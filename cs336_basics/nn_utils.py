import torch
from jaxtyping import Float


def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, "..."]:
    m = x.amax(dim, keepdim=True)
    x = (x - m).exp()
    x = x / x.sum(dim, keepdim=True)
    return x
