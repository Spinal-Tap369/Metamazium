# metamazium/snail_performer/trpo_fo.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical, kl_divergence
from torch.optim import Adam

# Basic Torch Utilities (Param-list based)
def flatten_params(param_list):
    """Flatten a list of parameters (Tensor) into a single 1D tensor."""
    return torch.cat([p.data.view(-1) for p in param_list])

def unflatten_params(flat_tensor, param_list):
    """
    Unflatten a 1D tensor into a list of tensors with the same shapes as in param_list.
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
    Set the given param_list Tensors using the data in flat_params (1D).
    """
    new_tensors = unflatten_params(flat_params, param_list)
    for p, new_p in zip(param_list, new_tensors):
        p.data.copy_(new_p)

def flat_grad(output, param_list, create_graph=False, retain_graph=False):
    """
    Compute and flatten gradients of output w.r.t. the parameters in param_list.
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


# Backtracking Line Search

def line_search(param_list, f, x, fullstep, expected_improve_rate, max_kl,
                max_backtracks=10, accept_ratio=0.1):
    """
    Backtracking line search to satisfy improvement and KL constraints.
    
    param_list: parameters to update.
    f: a callable that returns (loss, kl) (evaluated under no_grad).
    x: initial flattened parameters.
    fullstep: proposed update direction.
    expected_improve_rate: expected improvement along fullstep.
    max_kl: KL divergence constraint.
    """
    for stepfrac in [0.5 ** n for n in range(max_backtracks)]:
        xnew = x + stepfrac * fullstep
        set_params_flat(param_list, xnew)
        with torch.no_grad():
            loss, kl = f()
        actual_improve = -loss  # because loss is negative surrogate
        expected_improve = expected_improve_rate * stepfrac
        if (actual_improve / (expected_improve + 1e-8) > accept_ratio) and (kl < max_kl):
            return True, xnew
    return False, x

# TRPO_FO Class (First-Order TRPO using only first-order gradients)

class TRPO_FO:
    """
    A first-order TRPO implementation that uses only the first-order gradient.
    
    It computes the surrogate loss and its gradient, then uses the negative gradient
    as the update direction. A backtracking line search is performed to ensure that the KL divergence 
    between the new and old policies is below a threshold.
    
    This approximates TRPO without computing the natural gradient (i.e. without using 
    conjugate gradient and Hessian-vector products).
    
    References:
        - TRPO: https://arxiv.org/abs/1502.05477
        - RL^2: https://arxiv.org/pdf/1611.02779
    """
    def __init__(
        self,
        policy,
        value_fun,
        simulator=None,
        max_kl_div=0.01,
        discount=0.99,
        lam=0.95,
        vf_iters=5,
        max_value_step=0.01
    ):
        self.policy = policy
        self.value_fun = value_fun
        self.simulator = simulator
        self.max_kl_div = max_kl_div
        self.discount = discount
        self.lam = lam
        self.vf_iters = vf_iters
        self.max_value_step = max_value_step

        self.current_kl = 0.0

        self.policy_params = self.policy.policy_parameters()
        self.value_params = self.value_fun.value_parameters()

        self.value_optimizer = Adam(self.value_params, lr=self.max_value_step)

    def get_advantages(self, rollouts):
        all_advantages = []
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
            all_advantages.append(advs)
        return all_advantages

    def update_policy(self, states, actions, advantages, old_log_probs):
        """
        Perform a first-order policy update.
        
        Computes the surrogate loss and its gradient, uses the negative gradient as the update
        direction, and performs a backtracking line search to enforce the KL constraint.
        
        Returns:
            (policy_loss, dummy_value_loss=0.0)
        """
        old_params = flatten_params(self.policy_params)

        def get_loss_kl():
            dist = self._forward_dist(states)
            log_prob = dist.log_prob(actions)
            ratio = torch.exp(log_prob - old_log_probs)
            surr = ratio * advantages
            loss = -surr.mean()  # We minimize negative surrogate.
            with torch.no_grad():
                dist_old = Categorical(logits=dist.logits.detach())
            kl = torch.mean(kl_divergence(dist_old, dist))
            return loss, kl

        loss_old, kl_old = get_loss_kl()
        grad = flat_grad(loss_old, self.policy_params, create_graph=False, retain_graph=False)
        full_step = -grad  # first-order update direction (gradient descent step)

        exp_improve = -(grad * full_step).sum()

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
        logits, _, _ = self.policy.forward_with_state(
            states.unsqueeze(1),
            self._dummy_rnn_state(states.size(0))
        )
        logits = logits.squeeze(1)
        return Categorical(logits=logits)

    def _forward_value(self, states):
        _, values, _ = self.value_fun.forward_with_state(
            states.unsqueeze(1),
            self._dummy_rnn_state(states.size(0))
        )
        return values

    def _dummy_rnn_state(self, batch_size):
        num_layers = self.policy.num_layers
        hidden_size = self.policy.hidden_size
        device = next(self.policy.parameters()).device
        h = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (h, c)
