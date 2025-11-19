import torch
from cs336_basics.multiheadattention import MultiHeadAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rope import RoPE


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        x_tmp = self.ln1(x)
        x_tmp = self.attn(x_tmp, token_positions)
        x_attention = x_tmp + x
        x_tmp = self.ln2(x_attention)
        x_tmp = self.ffn(x_tmp)
        out = x_tmp + x_attention
        return out
