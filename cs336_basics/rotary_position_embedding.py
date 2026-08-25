import einx
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int


class RotaryPositionEmbedding(nn.Module):
    cos: Float[torch.Tensor, "max_seq_len d_half"]
    sin: Float[torch.Tensor, "max_seq_len d_half"]

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # $\theta ^ {-2k/d}$, k ∈ {0...d/2-1}; d = d_k
        inv_freq = theta ** (-torch.arange(0, d_k, 2, **factory_kwargs) / d_k)
        t = torch.arange(0, max_seq_len, **factory_kwargs)
        freqs = einx.dot("max_seq_len, d_half -> max_seq_len d_half", t, inv_freq)
        # cos/sin表
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def apply_rotate(
        x: Float[torch.Tensor, "... seq_len d"],
        cos: Float[torch.Tensor, "... seq_len d_half"],
        sin: Float[torch.Tensor, "... seq_len d_half"],
    ) -> Float[torch.Tensor, "... seq_len d"]:
        # 两两配对
        x0 = rearrange(x, "... (d_half two) -> ... d_half two", two=2)
        x1, x2 = x0[..., 0], x0[..., 1]
        y1 = cos * x1 - sin * x2
        y2 = sin * x1 + cos * x2
        y0 = torch.stack([y1, y2], dim=-1)
        y = rearrange(y0, "... d_half two -> ... (d_half two)", two=2)
        return y

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d"]:
        cos = einx.get_at("[max_seq_len] d_half, ... seq_len -> ... seq_len d_half", self.cos, token_positions)
        sin = einx.get_at("[max_seq_len] d_half, ... seq_len -> ... seq_len d_half", self.sin, token_positions)
        return self.apply_rotate(x, cos, sin)
