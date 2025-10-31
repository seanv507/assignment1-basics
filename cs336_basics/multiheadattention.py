
import einx
import torch
from cs336_basics.sdp_attention import Attention

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 device: torch.device | None=None, dtype: torch.dtype|None=None)-> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = int(self.d_model/self.num_heads)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.Q = torch.nn.Parameter(torch.empty(( d_model, d_model,), **factory_kwargs))
        self.K= torch.nn.Parameter(torch.empty(( d_model, d_model,), **factory_kwargs))
        self.V= torch.nn.Parameter(torch.empty(( d_model, d_model,), **factory_kwargs))
        self.O= torch.nn.Parameter(torch.empty(( d_model, d_model,), **factory_kwargs))
        self.sdp_attention = Attention()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        Q = einx.dot("b... s [d] , o  [d]->  b... s o", x, self.Q)
        K = einx.dot("b... s [d] , o  [d]->  b... s o", x, self.K)
        V = einx.dot("b... s [d] , o  [d]->  b... s o", x, self.V)
        Q_rearrange = einx.rearrange("b... s (h d) -> b... h s d", Q, h = self.num_heads)
        K_rearrange = einx.rearrange("b... s (h d) -> b... h s d", K, h = self.num_heads)
        V_rearrange = einx.rearrange("b... s (h d) -> b... h s d", V, h = self.num_heads)
        # As a stretch goal, try combining the key, query, and value projections into a single weight matrix so you only need a single
        # matrix multiply.

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones((seq_len, seq_len),dtype=torch.bool))
        att_out = self.sdp_attention(Q_rearrange, K_rearrange, V_rearrange, mask)
        att_out = einx.rearrange("b... h s d -> b... s (h d)", att_out, h = self.num_heads)
        
        out = einx.dot("b... s [d] , o  [d]->  b... s o", att_out, self.O)
        return out