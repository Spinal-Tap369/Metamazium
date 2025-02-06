# metamazium/lstm_ppo/trpo.py

"""
Trust Region Policy Optimization (TRPO) module.

This module implements the TRPO algorithm with Generalized Advantage
Estimation (GAE). It includes utilities for flattening parameters, computing
gradients and Hessian-vector products, solving linear systems via conjugate
gradient, and performing a backtracking line search to enforce a KL-divergence
constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical, kl_divergence
from torch.optim import Adam

# Basic Torch Utilities (param_list-based)
def flatten_params(param_list):
    """
    Flatten a list of tensors into a single 1D tensor.
    
    Args:
        param_list (list[torch.Tensor]): List of parameters.
    
    Returns:
        torch.Tensor: Flattened tensor.
    """
    return torch.cat([p.data.view(-1) for p in param_list])

def unflatten_params(flat_tensor, param_list):
    """
    Unflatten a 1D tensor into a list of tensors matching the shapes in param_list.
    
    Args:
        flat_tensor (torch.Tensor): Flattened tensor.
        param_list (list[torch.Tensor]): List of parameters whose shapes are desired.
    
    Returns:
        list[torch.Tensor]: List of tensors with corresponding shapes.
    """
    idx = 0
    new_params = []
    for p in param_list:
        numel = p.numel()
        vals = flat_tensor[idx: idx + numel]
        new_params.append(vals.view(p.shape))
        idx += numel
    return new_params

def set_params_flat(param_list, flat_params):
    """
    Update parameters in param_list with values from a flattened tensor.
    
    Args:
        param_list (list[torch.Tensor]): Parameters to update.
        flat_params (torch.Tensor): New values in flattened form.
    """
    new_tensors = unflatten_params(flat_params, param_list)
    for p, new_p in zip(param_list, new_tensors):
        p.data.copy_(new_p)

def flat_grad(output, param_list, create_graph=False, retain_graph=False):
    """
    Compute and flatten gradients of the output with respect to param_list.
    
    Args:
        output (torch.Tensor): Scalar output.
        param_list (list[torch.Tensor]): List of parameters.
        create_graph (bool): Whether to create a computational graph.
        retain_graph (bool): Whether to retain the graph.
    
    Returns:
        torch.Tensor: Flattened gradient vector.
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


# Conjugate Gradient Solver
def conjugate_gradient(Avp_func, b, max_iter=10, tol=1e-10):
    """
    Solve the linear system Ax = b using the conjugate gradient method,
    where Avp_func computes A*x.
    
    Args:
        Avp_func (callable): Function to compute A*x.
        b (torch.Tensor): Right-hand side vector.
        max_iter (int): Maximum iterations.
        tol (float): Tolerance for convergence.
    
    Returns:
        torch.Tensor: Approximate solution vector x.
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


# Hessian-Vector Product
def build_hessian_vector_product(loss, param_list, damping):
    """
    Build a function to compute Hessian-vector products plus a damping term.
    
    Args:
        loss (torch.Tensor): Scalar loss.
        param_list (list[torch.Tensor]): List of parameters.
        damping (float): Damping coefficient.
    
    Returns:
        tuple: (grads, hvp_func) where grads is the flattened gradient,
               and hvp_func(v) computes H*v + damping*v.
    """
    grads = flat_grad(loss, param_list, create_graph=True, retain_graph=True)
    
    def hvp_func(v):
        v = v.clone().detach().requires_grad_(True)
        with torch.backends.cudnn.flags(enabled=False):
            Hv = flat_grad((grads * v).sum(), param_list, retain_graph=True)
        return Hv + damping * v
    
    return grads, hvp_func


# Backtracking Line Search
def line_search(param_list, f, x, fullstep, expected_improve_rate, max_kl,
                max_backtracks=10, accept_ratio=0.1):
    """
    Perform backtracking line search to find a step that improves the objective
    while satisfying the KL divergence constraint.
    
    Args:
        param_list (list[torch.Tensor]): Parameters to update.
        f (callable): Function returning (loss, kl) with no gradient effects.
        x (torch.Tensor): Current flattened parameters.
        fullstep (torch.Tensor): Proposed update direction.
        expected_improve_rate (float): Expected improvement rate.
        max_kl (float): Maximum allowed KL divergence.
        max_backtracks (int): Maximum backtracking steps.
        accept_ratio (float): Minimum ratio of actual to expected improvement.
    
    Returns:
        tuple: (success (bool), new_flat_params (torch.Tensor))
    """
    for stepfrac in [0.5 ** n for n in range(max_backtracks)]:
        xnew = x + stepfrac * fullstep
        set_params_flat(param_list, xnew)
        
        with torch.no_grad():
            loss, kl = f()
        
        actual_improve = -loss  # Since we minimize -surrogate
        expected_improve = expected_improve_rate * stepfrac
        improve_ratio = actual_improve / (expected_improve + 1e-8)
        
        if (improve_ratio > accept_ratio) and (kl < max_kl):
            return True, xnew
    return False, x


# TRPO Class
class TRPO:
    """
    Trust Region Policy Optimization (TRPO) algorithm.

    This class implements TRPO with GAE, solving for the natural gradient
    direction using conjugate gradient, and applying a backtracking line search
    to ensure that the KL-divergence constraint is met.
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
        Initialize the TRPO trainer.

        Args:
            policy (nn.Module): LSTM policy network (producing distribution logits).
            value_fun (nn.Module): Value function network (can be shared with policy).
            simulator: Placeholder (unused).
            max_kl_div (float): Maximum KL divergence for trust region.
            discount (float): Discount factor (γ).
            lam (float): GAE lambda.
            cg_damping (float): Damping coefficient for Hessian-vector product.
            cg_max_iters (int): Maximum iterations for conjugate gradient.
            vf_iters (int): Number of iterations for value function updates.
            max_value_step (float): Learning rate for value function optimization.
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

        self.current_kl = 0.0

        # Retrieve parameter groups.
        self.policy_params = self.policy.policy_parameters()
        self.value_params = self.value_fun.value_parameters()

        self.value_optimizer = Adam(self.value_params, lr=self.max_value_step)

    def get_advantages(self, rollouts):
        """
        Compute Generalized Advantage Estimation (GAE) advantages for each rollout.

        Args:
            rollouts (list[dict]): Each dictionary must contain 'rewards' and 'values'.
        
        Returns:
            list[np.ndarray]: Advantage arrays corresponding to each rollout.
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
        Perform one TRPO policy update.

        Computes the surrogate loss and its gradient, uses conjugate gradient to
        approximate the natural gradient, scales the step to satisfy the KL constraint,
        and applies a backtracking line search.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions.
            advantages (torch.Tensor): Advantage estimates.
            old_log_probs (torch.Tensor): Log probabilities from the previous policy.
        
        Returns:
            tuple: (policy_loss, dummy_value_loss), where dummy_value_loss is zero.
        """
        old_params = flatten_params(self.policy_params)

        def get_loss_kl():
            dist = self._forward_dist(states)
            log_prob = dist.log_prob(actions)
            ratio = torch.exp(log_prob - old_log_probs)
            surr = ratio * advantages
            loss = -surr.mean()
            with torch.no_grad():
                dist_old = Categorical(logits=dist.logits.detach())
            kl = torch.mean(kl_divergence(dist_old, dist))
            return loss, kl

        loss_old, _ = get_loss_kl()

        grads, hvp_func = build_hessian_vector_product(loss_old, self.policy_params, self.cg_damping)
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

        return -final_loss, 0.0

    def update_value_fun(self, states, target_values):
        """
        Update the value function via MSE loss over a number of iterations.

        Args:
            states (torch.Tensor): Batch of states.
            target_values (torch.Tensor): Target return values.
        
        Returns:
            float: Final MSE loss after updating.
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

    def _forward_dist(self, states):
        """
        Compute the action distribution from the policy network.

        Args:
            states (torch.Tensor): Batch of states of shape [T, ...].
        
        Returns:
            torch.distributions.Categorical: Distribution over actions.
        """
        logits, _, _ = self.policy.forward_with_state(
            states.unsqueeze(1),
            self._dummy_rnn_state(states.size(0))
        )
        logits = logits.squeeze(1)
        return Categorical(logits=logits)

    def _forward_value(self, states):
        """
        Compute value estimates from the value network.

        Args:
            states (torch.Tensor): Batch of states of shape [T, ...].
        
        Returns:
            torch.Tensor: Value estimates.
        """
        _, values, _ = self.value_fun.forward_with_state(
            states.unsqueeze(1),
            self._dummy_rnn_state(states.size(0))
        )
        return values

    def _dummy_rnn_state(self, batch_size):
        """
        Create a dummy (zero) RNN state for the given batch size.

        Args:
            batch_size (int): Batch size.
        
        Returns:
            tuple: Zero tensors for hidden and cell states.
        """
        num_layers = self.policy.num_layers
        hidden_size = self.policy.hidden_size
        device = next(self.policy.parameters()).device
        h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return h, c
