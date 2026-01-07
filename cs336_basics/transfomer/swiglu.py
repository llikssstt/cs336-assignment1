import torch
import torch.nn as nn
from einops import rearrange, einsum

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dtype=None, device=None) -> None:
        super(SwiGLU, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype)).to(device=device)
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), dtype=dtype)).to(device=device)
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), dtype=dtype)).to(device=device)

        std = (2 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(self.w1, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w1x = einsum(x, self.w1, '... d_model, d_ff d_model -> ... d_ff')
        w3x = einsum(x, self.w3, '... d_model, d_ff d_model -> ... d_ff')
        a = w1x * self.sigmoid(w1x) * w3x
        b = einsum(self.w2, a, 'd_model d_ff, ... d_ff -> ... d_model')
        return b

