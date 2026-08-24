import einx
import torch
import torch.nn as nn
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        weight = torch.ones(d_model, **factory_kwargs)
        self.weight = nn.Parameter(weight)  # learnable gain
        self.eps = eps

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = (einx.mean("... [d_model]", x**2) + self.eps) ** 0.5
        result = einx.multiply("... d_model, ..., d_model -> ... d_model", x, 1 / rms, self.weight)
        return result.to(in_type)
