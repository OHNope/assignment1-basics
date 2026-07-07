import torch
import torch.nn as nn


# adopt the dynamic calc of the cos/ sin
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        positions: torch.Tensor = torch.arange(max_seq_len, device=device)
        dims: torch.Tensor = torch.arange(0, d_k, 2, device=device)

        inv_freq: torch.Tensor = 1.0 / (theta ** (dims / d_k))

        angles: torch.Tensor = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
        # max_seq_len, d_k

    def forward(
        self,
        x: torch.Tensor,  # (..., seq_len, d_k)
        token_positions: torch.Tensor,  # (..., seq_len)
    ) -> torch.Tensor:  # same as x
        with torch.cuda.nvtx.range("rope"):
            cos = self.cos_cached[token_positions]
            sin = self.sin_cached[token_positions]

            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]

            out_even = cos * x_even - sin * x_odd
            out_odd = sin * x_even + cos * x_odd

            out = torch.empty_like(x)
            out[..., 0::2] = out_even
            out[..., 1::2] = out_odd

            return out
        # use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
