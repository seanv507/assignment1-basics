import math
import torch
from einx import dot
from cs336_basics.softmax import SoftMax


class Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax = SoftMax(-1)

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """given x  and weights"""
        qt_k = dot("b... a [d], b... c [d] -> b... a c", Q, K)
        d_k = K.size(-1)
        qt_k /= math.sqrt(d_k)
        if mask is not None:
            qt_k.masked_fill_(~mask, -torch.inf)
        attention = self.softmax(qt_k)
        out = dot("b... h s1 [s2], b... h [s2] d_out-> b... h s1 d_out ", attention, V)
        return out
