import torch
from torch import Tensor

def run_silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)