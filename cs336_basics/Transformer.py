import torch
import torch.nn as nn

from .Normalizaiton import RMSNorm
from .Attention import MHA
from .Linear import SwiGLU, Linear
from .Softmax import softmax
from .Embedding import Embedding

# one transformer block

# sub layers : 1. MHA 2. SwiGLU feed-forward network

# layer : RMSNorm + (MHA/FF) + residual networks


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.attn: MHA = MHA(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ln1: RMSNorm = RMSNorm(d_model)
        self.ffn: SwiGLU = SwiGLU(d_model, d_ff)
        self.ln2: RMSNorm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor  # (batch_size, seq_len, d_model)
    ) -> torch.Tensor:  # (batch_size, seq_len, d_model)
        # Add1: x + (Norm + MHA(RoPE))
        # Add2: Add1 + (Norm +Position-Wise feed-forward)

        # y = x + MultiHeadSelfAttention(RMSNorm(x)).

        with torch.cuda.nvtx.range("transformer_block_attn"):
            y = x + (self.attn.forward_with_rope(self.ln1.forward(x)))

        with torch.cuda.nvtx.range("transformer_block_ffn"):
            z = y + (self.ffn.forward(self.ln2.forward(y)))

        return z


# num_layers = num of the Transformer Blocks

"""
Token embedding 

Transformer blocks ...

Norm 

Lineae 

Softmax 

Out put probability (unnormalized ANs)


"""


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
    ):
        super().__init__()

        # voac size = nums of the embeddings

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("embedding"):
            x = self.token_embeddings.forward(in_indices)

        for i, layer in enumerate(self.layers):
            with torch.cuda.nvtx.range(f"transformer_layer_{i}"):
                x = layer(x)

        with torch.cuda.nvtx.range("final_norm"):
            x = self.ln_final.forward(x)

        with torch.cuda.nvtx.range("lm_head"):
            x = self.lm_head.forward(x)

        return x  # no softmax, unnormalized answer
