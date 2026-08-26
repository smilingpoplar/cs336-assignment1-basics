import einx
import torch
from jaxtyping import Bool, Float


def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    m = x.amax(dim, keepdim=True)
    x = (x - m).exp()
    x = x / x.sum(dim, keepdim=True)
    return x


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "... seq_q d_k"],
    K: Float[torch.Tensor, "... seq_k d_k"],
    V: Float[torch.Tensor, "... seq_k d_v"],
    mask: Bool[torch.Tensor, "seq_q seq_k"] | None = None,
) -> Float[torch.Tensor, "... seq_q d_v"]:
    qk = einx.dot("... seq_q [d_k], ... seq_k [d_k] -> ... seq_q seq_k", Q, K)
    qk = qk / K.shape[-1] ** 0.5
    if mask is not None:
        qk.masked_fill_(~mask, -1e9)
    a = softmax(qk, dim=-1)
    return einx.dot("... seq_q [seq_k], ... [seq_k] d_v -> ... seq_q d_v", a, V)
