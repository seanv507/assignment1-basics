import torch
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.empty(
                (
                    out_features,
                    in_features,
                ),
                **factory_kwargs,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = (2 / (self.weight.size(0) + self.weight.size(1))) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
