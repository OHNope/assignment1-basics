from __future__ import annotations

try:
    from .Embedding import Embedding
    from .Linear import Linear, SwiGLU
    from .Normalizaiton import RMSNorm
    from .Transformer import Transformer as BasicsTransformerLM
except ImportError:
    from .model import BasicsTransformerLM, Embedding, Linear, RMSNorm, SwiGLU

# export the needed parts
__all__ = ["BasicsTransformerLM", "Embedding", "Linear", "RMSNorm", "SwiGLU"]

