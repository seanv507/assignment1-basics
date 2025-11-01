import numpy as np
import torch


def get_batch(
    token_ids: np.array, batch_size: int, context_length: int, device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    n_points = max(len(token_ids) - context_length, 0)
    rng = np.random.default_rng()
    samples = rng.choice(n_points, batch_size, replace=True)
    X = torch.tensor([token_ids[i_token_id : i_token_id + context_length] for i_token_id in samples], device=device)
    y = torch.tensor(
        [token_ids[i_token_id + 1 : i_token_id + context_length + 1] for i_token_id in samples], device=device
    )
    return (X, y)


def load_data(file):
    data = np.memmap(file, mode="r")
    return data
