import torch
import math
from collections.abc import Callable


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas, eps: float, weight_decay: float):
        defaults = {"alpha": lr, "beta_1": betas[0], "beta_2": betas[1], "epsilon": eps, "lamda": weight_decay}
        # whgat does params do?
        # and defaults?

        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            epsilon = group["epsilon"]
            lamda = group["lamda"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["exp_avg"] += (1 - beta_1) * (p.grad - state["exp_avg"])
                state["exp_avg_sq"] += (1 - beta_2) * (p.grad**2 - state["exp_avg_sq"])
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                # start at 1 to avoid 0/0
                alpha_t = alpha * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= (
                    alpha_t * state["exp_avg"] / (torch.sqrt(state["exp_avg_sq"]) + epsilon)
                )  # Update weight tensor in-place.
                p.data *= 1 - alpha * lamda
                state["t"] = t + 1  # Increment iteration number.
        return loss
