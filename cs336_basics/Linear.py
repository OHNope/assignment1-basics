import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()  # call super
        weight_tensor: torch.Tensor = torch.empty(
            out_features, in_features, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(weight_tensor)
        self.weight = nn.Parameter(weight_tensor)  # turn tensor into the real params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("linear"):
            return x @ self.weight.T  # apply linear transformation to input


# x:        (..., in_features)
# weight:   (out_features, in_features) Normal `Wx` mode
# weight.T: (in_features, out_features)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)  # (out, in) mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("swiglu_gate_up"):
            x1 = self.w1(x)
            x3 = self.w3(x)

        with torch.cuda.nvtx.range("swiglu_activation"):
            hidden = (x1 * torch.sigmoid(x1)) * x3

        with torch.cuda.nvtx.range("swiglu_down"):
            return self.w2(hidden)
