import math
import torch
import einx


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        max_dim_pair = int(d_k // 2)
        rotations = torch.empty((max_seq_len, max_dim_pair, 2, 2), device=device)

        for dim_pair in range(max_dim_pair):
            theta_step = theta ** (-dim_pair / max_dim_pair)
            theta_ik = 0
            for position in range(max_seq_len):
                rotations[position, dim_pair, :, :] = torch.Tensor(
                    [[math.cos(theta_ik), -math.sin(theta_ik)], [math.sin(theta_ik), math.cos(theta_ik)]]
                )
                theta_ik += theta_step

        self.register_buffer("rotations", rotations, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        batch_len, seq_len, embed_len = x.size()
        rotations_lookup = torch.index_select(self.rotations, 0, token_positions)
        dim_pair = int(embed_len // 2)
        x_reshape = x.reshape((batch_len, seq_len, dim_pair, 2, 1))
        rotated_x = torch.matmul(rotations_lookup, x_reshape)
        out = rotated_x.reshape((batch_len, seq_len, embed_len))
        return out
