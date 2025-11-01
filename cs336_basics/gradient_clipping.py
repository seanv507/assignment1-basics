import torch


def clip_gradients_(parameters, M, epsilon=1e-6):
    breakpoint()
    for parameter in parameters:
        if (grad := parameter.grad) is None:
            continue
        norm = torch.linalg.vector_norm(grad)
        if norm >= M:
            grad.data *= M / (norm + epsilon)
