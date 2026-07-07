import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Softmax import softmax
from .RoPE import RotaryPositionalEmbedding
from .Linear import Linear


def scaled_dot_product_attention_no_mask(
    Q: torch.Tensor,  # (..., seq_len, d_k)
    K: torch.Tensor,  # (..., seq_len, d_k)
    V: torch.Tensor,  # (..., seq_len, d_v)
) -> torch.Tensor:  # (..., seq_len, d_v)
    with torch.cuda.nvtx.range("sdpa_no_mask"):
        return F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)


def scaled_dot_product_attention(
    Q: torch.Tensor,  # (..., seq_len, d_k)
    K: torch.Tensor,  # (..., seq_len, d_k)
    V: torch.Tensor,  # (..., seq_len, d_v)
    mask: torch.Tensor | None = None,  # (..., seq_len, seq_len) or (seq_len, seq_len)
    is_causal: bool = False,
) -> torch.Tensor:  # (..., seq_len, d_v)
    d_k = Q.shape[-1]

    with torch.cuda.nvtx.range("attn_qk_matmul"):
        scores = Q @ K.transpose(-2, -1)

    with torch.cuda.nvtx.range("attn_scale_mask"):
        scores = scores / math.sqrt(d_k)

        if is_causal:
            q_len = Q.shape[-2]
            k_len = K.shape[-2]
            causal_mask = torch.ones(
                q_len,
                k_len,
                dtype=torch.bool,
                device=Q.device,
            ).tril(diagonal=k_len - q_len)
            scores = scores.masked_fill(~causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

    with torch.cuda.nvtx.range("attn_softmax"):
        attn = softmax(scores, dim=-1)

    with torch.cuda.nvtx.range("attn_av_matmul"):
        return attn @ V


class MHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()

        self.h = num_heads
        self.d_model = d_model
        self.d_k = self.d_v = self.d_model // self.h  # flooring division

        if (max_seq_len is not None) & (theta is not None):
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)

        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.output_proj = Linear(num_heads * self.d_v, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (..., seq_len, d_k)
        # x: (..., seq_len, d_model)

        seq_len = x.shape[-2]

        with torch.cuda.nvtx.range("mha_qkv_proj"):
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

        with torch.cuda.nvtx.range("mha_reshape"):
            # MultiHeads : with no change of the params,increase expression
            # (..., seq_len, d_model) -> (..., seq_len, h, d_k)
            Q = Q.view(*Q.shape[:-1], self.h, self.d_k)
            K = K.view(*K.shape[:-1], self.h, self.d_k)
            V = V.view(*V.shape[:-1], self.h, self.d_v)
            # (..., seq_len, num_heads, d_k)  -> (..., num_heads, seq_len, d_k)

            Q = Q.transpose(-3, -2)
            K = K.transpose(-3, -2)
            V = V.transpose(-3, -2)

        with torch.cuda.nvtx.range("mha_attention"):
            attn_out = scaled_dot_product_attention(Q, K, V, is_causal=True)

        with torch.cuda.nvtx.range("mha_output_proj"):
            # (..., num_heads, seq_len, d_k) -> (..., seq_len, num_heads, d_k)
            attn_out = attn_out.transpose(-3, -2)

            attn_out = attn_out.contiguous().view(*x.shape[:-1], self.d_model)

            return self.output_proj(attn_out)

    def forward_with_rope(
        self,
        x: torch.Tensor,  # (..., seq_len, d_model)
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)

        with torch.cuda.nvtx.range("mha_qkv_proj"):
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

        with torch.cuda.nvtx.range("mha_reshape"):
            # MultiHeads : with no change of the params,increase expression
            # (..., seq_len, d_model) -> (..., seq_len, h, d_k)
            Q = Q.view(*Q.shape[:-1], self.h, self.d_k)
            K = K.view(*K.shape[:-1], self.h, self.d_k)
            V = V.view(*V.shape[:-1], self.h, self.d_v)
            # (..., seq_len, num_heads, d_k)  -> (..., num_heads, seq_len, d_k)

            Q = Q.transpose(-3, -2)
            K = K.transpose(-3, -2)
            V = V.transpose(-3, -2)

        with torch.cuda.nvtx.range("mha_rope"):
            # RoPE
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        with torch.cuda.nvtx.range("mha_attention"):
            attn_out = scaled_dot_product_attention(Q, K, V, is_causal=True)

        with torch.cuda.nvtx.range("mha_output_proj"):
            # (..., num_heads, seq_len, d_k) -> (..., seq_len, num_heads, d_k)
            attn_out = attn_out.transpose(-3, -2)

            attn_out = attn_out.contiguous().view(*x.shape[:-1], self.d_model)

            return self.output_proj(attn_out)

    # torch.tril : excerpt the lower triangular part of the matrices
    # calc dim
    # do attn for every head
    # concat the heads
    # apply the rope
    # causual masking


# TODO: why adding a -\infty works well
