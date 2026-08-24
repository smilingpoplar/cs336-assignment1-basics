import math

import einx
import torch
import torch.nn as nn
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,  # d_in
        out_features: int,  # d_out
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        weight = torch.empty((out_features, in_features), **factory_kwargs)
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(weight)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        return einx.dot("... [d_in], d_out [d_in] -> ... d_out", x, self.weight)
