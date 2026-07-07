import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_model, device=device, dtype=dtype))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("rms_norm"):
            in_dtype = x.dtype

            x = x.to(torch.float32)

            rms = torch.sqrt(
                torch.mean(x * x, dim=-1, keepdim=True) + self.eps
            )  # dim = -1 : along the last dim = d_model to calc the mean

            result = x / rms * self.weight.to(torch.float32)

            return result.to(in_dtype)
