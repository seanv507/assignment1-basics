import os
import typing
import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration, 
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoints = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoints, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer)-> int:
    checkpoints = torch.load(src)
    model.load_state_dict(checkpoints["model"])
    optimizer.load_state_dict(checkpoints["optimizer"])
    iteration = checkpoints["iteration"]
    return iteration

