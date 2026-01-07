import torch
import torch.nn as nn
from einops import einsum

def Softmax(x: torch.Tensor, dimension: int) -> torch.Tensor:
    x = x.to(dtype=torch.float32)

    max_v = torch.max(x, dim=dimension, keepdim=True).values

    new_x = torch.exp(x - max_v)

    sum_d = torch.sum(new_x, dim=dimension, keepdim=True)

    norm_x = new_x / sum_d
    return norm_x

