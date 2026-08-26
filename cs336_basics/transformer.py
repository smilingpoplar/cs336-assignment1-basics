import torch
import torch.nn as nn
from jaxtyping import Float, Int

from .embedding import Embedding
from .linear import Linear
from .multihead_self_attention import MultiheadSelfAttention
from .rmsnorm import RMSNorm
from .rotary_position_embedding import RotaryPositionEmbedding
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        rope = RotaryPositionEmbedding(theta, d_model // num_heads, max_seq_len, **factory_kwargs)
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> Float[torch.Tensor, "batch seq_len d_model"]:
        seq_len = x.shape[-2]
        x += self.attn(self.ln1(x), token_positions=torch.arange(0, seq_len))
        x += self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **factory_kwargs)
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self, in_indices: Int[torch.Tensor, "batch seq_len"]
    ) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
