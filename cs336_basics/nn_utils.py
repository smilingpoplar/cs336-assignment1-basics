import torch
from jaxtyping import Float


def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)
