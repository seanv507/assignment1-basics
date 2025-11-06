import torch
from einx import get_at


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embeddings_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(Embedding, self).__init__()
        # subclass nn.Module
        # call the superclass constructor
        # initialize your embedding matrix as a nn.Parameter
        # store the embedding matrix with the d_model being the final dimension
        self.num_embeddings = num_embeddings
        self.embeddings_dim = embeddings_dim
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = torch.nn.Parameter(
            torch.empty(
                (
                    num_embeddings,
                    embeddings_dim,
                ),
                **factory_kwargs,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = (2 / (self.weight.size(0) + self.weight.size(1))) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """given x  and weights"""
        out = get_at("[i] d, b...  -> b... d", self.weight, x)
        # out = self.weight[x, :]
        return out

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embeddings_dim={self.embeddings_dim}"
