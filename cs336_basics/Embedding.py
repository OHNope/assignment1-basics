import torch
import torch.nn as nn


class Embedding(nn.Module):  # Inputs token -> Embedding spaces
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()  # call super

        embedding_tensor: torch.Tensor = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(embedding_tensor)
        )  # turn tensor into the real params

    def forward(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor:  # Look up method of the embedding space
        return self.weight[token_ids]


"""
token_ids: (batch, seq)
  weight:    (vocab_size, d_model)

  output:    (batch, seq, d_model)
"""
