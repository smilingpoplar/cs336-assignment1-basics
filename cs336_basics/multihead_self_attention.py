import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from .linear import Linear
from .nn_utils import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"]) -> Float[torch.Tensor, "... seq_len d_model"]:
        q = rearrange(self.q_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        attn: Float[torch.Tensor, "... h seq_len d_v"] = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "... h seq_len d_v -> ... seq_len (h d_v)", h=self.num_heads)
        return self.output_proj(attn)
