# metamazium/lstm_ppo/trpo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical, kl_divergence
from torch.optim import Adam

##############################################################################
# Basic Torch Utilities (Now param_list-based)
##############################################################################

def flatten_params(param_list):
    """Flatten a list of parameters (Tensor) into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in param_list])

def unflatten_params(flat_tensor, param_list):
    """
    Unflatten 1D `flat_tensor` into the shapes of param_list.
    Return a list of Tensors with the same shapes as in param_list.
    """
    idx = 0
    new_params = []
    for p in param_list:
        numel = p.numel()
        vals = flat_tensor[idx : idx + numel]
        new_params.append(vals.view(p.shape))
        idx += numel
    return new_params

def set_params_flat(param_list, flat_params):
    """
    Set the given param_list Tensors using the data in flat_params (1D).
    """
    new_tensors = unflatten_params(flat_params, param_list)
    for p, new_p in zip(param_list, new_tensors):
        p.data.copy_(new_p)

def flat_grad(output, param_list, create_graph=False, retain_graph=False):
    """
    Compute and flatten gradients of `output` w.r.t. the param_list.
    allow_unused=True to avoid errors if some params are not used in the graph.
    """
    grads = torch.autograd.grad(
        output, param_list,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True
    )
    out = []
    for g, p in zip(grads, param_list):
        if g is None:
            out.append(torch.zeros_like(p).view(-1))
        else:
            out.append(g.reshape(-1))
    return torch.cat(out)

##############################################################################
# Conjugate Gradient Solver
##############################################################################

def conjugate_gradient(Avp_func, b, max_iter=10, tol=1e-10):
    """
    Solve Ax = b using Conjugate Gradient, 
    where Avp_func(x) = A*x (the Hessian-vector product).
    b is the first-order gradient (already flattened).
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)

    for _ in range(max_iter):
        Avp = Avp_func(p)
        alpha = rr / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        rr_new = torch.dot(r, r)
        if rr_new < tol:
            break
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    return x

##############################################################################
# Hessian-Vector Product
##############################################################################

def build_hessian_vector_product(loss, param_list, damping):
    """
    Returns (grads, hvp_func):
      grads: first-order gradient of 'loss' w.r.t. param_list
      hvp_func: function that takes 'v' -> H·v + damping·v
    """
    # Compute first-order gradients with graph retention.
    grads = flat_grad(loss, param_list, create_graph=True, retain_graph=True)

    def hvp_func(v):
        # To avoid in-place modifications, clone and detach v, then re-enable gradients.
        v = v.clone().detach().requires_grad_(True)
        # Compute (grads * v).sum() and then its gradient w.r.t. param_list.
        Hv = flat_grad((grads * v).sum(), param_list, retain_graph=True)
        return Hv + damping * v

    return grads, hvp_func

##############################################################################
# Backtracking Line Search
##############################################################################

def line_search(param_list, f, x, fullstep, expected_improve_rate, max_kl,
                max_backtracks=10, accept_ratio=0.1):
    """
    Backtracking line search to ensure we satisfy improvement & KL constraints.
    param_list: the parameters to update in place.
    f: a callable that returns (loss, kl) with no gradient side-effects.
       f() is run inside a no_grad block.
    x: the initial flattened parameters.
    fullstep: proposed direction from CG.
    expected_improve_rate: to measure improvement ratio.
    max_kl: KL constraint.
    """
    for stepfrac in [0.5 ** n for n in range(max_backtracks)]:
        xnew = x + stepfrac * fullstep
        set_params_flat(param_list, xnew)

        with torch.no_grad():
            loss, kl = f()

        actual_improve = -loss  # negative of new_loss (since we minimize -surrogate)
        expected_improve = expected_improve_rate * stepfrac
        improve_ratio = actual_improve / (expected_improve + 1e-8)

        if (improve_ratio > accept_ratio) and (kl < max_kl):
            return True, xnew

    return False, x

##############################################################################
# TRPO Class
##############################################################################

class TRPO:
    """
    Minimal TRPO that:
      - Computes GAE-based advantages (get_advantages)
      - Performs one TRPO policy update (update_policy)
      - Updates the value function (update_value_fun)
      - Logs current KL for reference.
    """

    def __init__(
        self,
        policy,
        value_fun,
        simulator=None,
        max_kl_div=0.01,
        discount=0.99,
        lam=0.95,
        cg_damping=0.1,
        cg_max_iters=10,
        vf_iters=5,
        max_value_step=0.01
    ):
        """
        Args:
            policy (nn.Module): LSTM policy network (outputs distribution logits).
            value_fun (nn.Module): The same or separate network's value head.
            simulator: Unused placeholder.
            max_kl_div: KL divergence limit for trust region.
            discount: Gamma for rewards.
            lam: GAE lambda.
            cg_damping: Damping for Hessian-vector product.
            cg_max_iters: Conjugate gradient iterations.
            vf_iters: Number of optimization steps for the value function.
            max_value_step: Learning rate for the value function.
        """
        self.policy = policy
        self.value_fun = value_fun
        self.simulator = simulator
        self.max_kl_div = max_kl_div
        self.discount = discount
        self.lam = lam
        self.cg_damping = cg_damping
        self.cg_max_iters = cg_max_iters
        self.vf_iters = vf_iters
        self.max_value_step = max_value_step

        self.current_kl = 0.0  # For logging

        # Separate parameter groups.
        self.policy_params = self.policy.policy_parameters()
        self.value_params = self.value_fun.value_parameters()

        self.value_optimizer = Adam(self.value_params, lr=self.max_value_step)

    def get_advantages(self, rollouts):
        """
        Compute GAE advantages from rollout data.
        Each rollout is a dict with keys: 'rewards' and 'values'.
        Returns a list of advantage arrays (one per rollout).
        """
        all_advs = []
        for data in rollouts:
            rewards = data['rewards']
            values = data['values']
            T = len(rewards)
            advs = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values[t+1] if t < T - 1 else 0.0
                delta = rewards[t] + self.discount * next_val - values[t]
                gae = delta + self.discount * self.lam * gae
                advs[t] = gae
            all_advs.append(advs)
        return all_advs

    def update_policy(self, states, actions, advantages, old_log_probs):
        """
        Perform one TRPO update:
          - Compute the surrogate loss and its gradient.
          - Build a Hessian-vector product function.
          - Use conjugate gradient to get a search direction.
          - Use line search to determine a step size.
        Returns (policy_loss, dummy_value_loss=0.0).
        """
        old_params = flatten_params(self.policy_params)

        def get_loss_kl():
            dist = self._forward_dist(states)
            log_prob = dist.log_prob(actions)
            ratio = torch.exp(log_prob - old_log_probs)
            surr = ratio * advantages
            loss = -surr.mean()  # minimize negative surrogate
            with torch.no_grad():
                dist_old = Categorical(logits=dist.logits.detach())
            kl = torch.mean(kl_divergence(dist_old, dist))
            return loss, kl

        loss_old, kl_old = get_loss_kl()

        # Build first-order gradients and HVP function (single backward pass)
        grads, hvp_func = build_hessian_vector_product(loss_old, self.policy_params, self.cg_damping)

        # Solve for search direction using conjugate gradient.
        search_dir = conjugate_gradient(hvp_func, grads, max_iter=self.cg_max_iters)

        shs = (search_dir * hvp_func(search_dir)).sum()
        step_scale = torch.sqrt(2.0 * self.max_kl_div / (shs + 1e-8))
        full_step = search_dir * step_scale

        exp_improve = -(grads * full_step).sum()

        def line_search_loss_kl():
            with torch.no_grad():
                new_loss, new_kl = get_loss_kl()
            return new_loss.item(), new_kl.item()

        success, new_params = line_search(
            param_list=self.policy_params,
            f=line_search_loss_kl,
            x=old_params,
            fullstep=full_step,
            expected_improve_rate=exp_improve,
            max_kl=self.max_kl_div
        )

        if not success:
            set_params_flat(self.policy_params, old_params)

        final_loss, final_kl = line_search_loss_kl()
        self.current_kl = final_kl

        return (-final_loss, 0.0)

    def update_value_fun(self, states, target_values):
        """
        Update the value function using MSE loss over vf_iters iterations.
        """
        value_loss = 0.0
        for _ in range(self.vf_iters):
            self.value_optimizer.zero_grad()
            predicted = self._forward_value(states).squeeze(-1)
            loss = F.mse_loss(predicted, target_values)
            loss.backward()
            self.value_optimizer.step()
            value_loss = loss.item()
        return value_loss

    # ------------------- Internal Helpers ------------------- #

    def _forward_dist(self, states):
        """
        Forward pass to obtain a Categorical distribution over actions.
        Expects states of shape [T, ...]. Uses policy.forward_with_state.
        """
        logits, _, _ = self.policy.forward_with_state(
            states.unsqueeze(1),
            self._dummy_rnn_state(states.size(0))
        )
        logits = logits.squeeze(1)
        return Categorical(logits=logits)

    def _forward_value(self, states):
        """
        Forward pass for the value function.
        """
        _, values, _ = self.value_fun.forward_with_state(
            states.unsqueeze(1),
            self._dummy_rnn_state(states.size(0))
        )
        return values

    def _dummy_rnn_state(self, batch_size):
        """
        Returns zero hidden and cell states.
        """
        num_layers = self.policy.num_layers
        hidden_size = self.policy.hidden_size
        device = next(self.policy.parameters()).device
        h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (h, c)
