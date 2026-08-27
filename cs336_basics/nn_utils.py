import einx
import torch
from jaxtyping import Bool, Float, Int


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


def cross_entropy(
    inputs: Float[torch.Tensor, "batch_size vocab_size"],
    targets: Int[torch.Tensor, " batch_size"],
) -> Float[torch.Tensor, ""]:
    # log(softmax(x))的写法，会让softmax(偏离最大x值较远的x项)下溢为0
    # logsumexp写法：
    batch_size = inputs.shape[-2]
    target_logits = inputs[torch.arange(batch_size), targets]
    m = inputs.amax(dim=-1, keepdim=True)
    log_sum_exp = (inputs - m).exp().sum(dim=-1).log() + m
    return (log_sum_exp - target_logits).mean()
