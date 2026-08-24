import torch
import torch.nn as nn
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # vocab_size (d_in)
        embedding_dim: int,  # d_model (d_out)
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        weight = torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        return self.weight[token_ids, :]
