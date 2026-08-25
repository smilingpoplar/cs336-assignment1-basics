import torch
import torch.nn as nn
from jaxtyping import Float

from .linear import Linear
from .nn_utils import silu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)  # gate_proj
        self.w2 = Linear(d_ff, d_model, **factory_kwargs)  # down_proj
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)  # up_proj

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return self.w2(silu(self.w1(x)) * self.w3(x))
