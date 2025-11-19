import torch
from cs336_basics.rope import RoPE
from cs336_basics.transformer import Transformer
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.softmax import SoftMax


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        d_k = int(d_model / num_heads)
        self.rope = RoPE(theta, d_k, context_length, device)
        self.transformers = torch.nn.ModuleList([
            Transformer(d_model, num_heads, d_ff, self.rope, device, dtype) for _ in range(num_layers)
        ])

        self.norm_output = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear_output = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.softmax = SoftMax(-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        out = self.embedding(x)
        for transformer in self.transformers:
            out=transformer(out, token_positions)
        
        out = self.norm_output(out)
        out = self.linear_output(out)
        return out
