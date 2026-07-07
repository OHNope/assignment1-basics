import torch
from collections.abc import Callable, Iterable
from typing import Optional

import math


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(
        -1
    )

    return (torch.logsumexp(inputs, dim=-1) - target_logits).mean()

    # Take the mean of the multiple outputs
    # logP (class = x_{i+1} | o_i) o_i = logits, describe the original scores of every possibel tokens

    # use softmax to normalize the whole scores


# perplexity = exp( 1/𝑚 ∑𝑚𝑖=1 ℓ𝑖).


# 𝜃𝑡+1 ← 𝜃𝑡− 𝛼𝑡∇𝐿(𝜃𝑡; 𝐵𝑡),
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


"""
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e2)
for t in range(100):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # Compute a scalar loss value.
    print(loss.cpu().item())
    30
    loss.backward()  # Run backward pass, which computes gradients.
    opt.step()  # Run optimizer step.
"""


# Support for the multi groups(multi params for every group) optimization
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        with torch.no_grad(): # evaluating mode 
            for group in self.param_groups: # contains components like : transformer, linear, ...
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for theta in group["params"]:
                    if theta.grad is None:
                        continue

                    grad = (
                        theta.grad
                    )  # get the gradients of the loss function related to the vriable theta
                    state = self.state[
                        theta
                    ]  # get the state related the theta variable

                    if len(state) == 0:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(theta)
                        state["v"] = torch.zeros_like(theta)

                    m = state["m"]
                    v = state["v"]

                    state["step"] += 1
                    t = state["step"]

                    alpha_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)

                    theta.add_(theta, alpha=-lr * weight_decay)

                    # m <- beta1 * m + (1 - beta1) * grad
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # v <- beta2 * v + (1 - beta2) * grad^2
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    theta.addcdiv_(m, v.sqrt() + eps, value=-alpha_t)

        return loss


# TODO:adamw_accounting


def learning_rate_schedule(
    t: int,
    a_max: float,
    a_min: float,
    T_w: int,
    T_c: int,
) -> float:  # return the rescheduled learning rate to use
    if t < T_w:
        return a_max * t / T_w
    if t <= T_c:
        return a_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (
            a_max - a_min
        )
    return a_min


eps: float = 1e-6


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return

    total_norm_sq = torch.zeros((), device=grads[0].device)

    for grad in grads:
        total_norm_sq += torch.sum(grad.detach() ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.mul_(scale)

