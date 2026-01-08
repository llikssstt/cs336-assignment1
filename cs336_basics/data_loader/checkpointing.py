from typing import Union, BinaryIO, IO
import os
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    """
    Save a training checkpoint to a path or file-like object.

    Stores:
      - model_state: model.state_dict()
      - optimizer_state: optimizer.state_dict()
      - iteration: current iteration (int)

    Args:
        model: torch.nn.Module
        optimizer: torch.optim.Optimizer
        iteration: int
        out: path-like or a binary file-like object opened for writing.
    """
    if not isinstance(iteration, int):
        raise TypeError(f"iteration must be int, got {type(iteration)}")

    # Package all required states into one object
    obj = {
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    # torch.save supports both file paths and file-like objects
    torch.save(obj, out)

def load_checkpoint(src, model, optimizer) -> int:
    ckpt = torch.load(src)
    iteration = ckpt['iteration']
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return iteration