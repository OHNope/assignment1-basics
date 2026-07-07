import torch


def softmax(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    with torch.cuda.nvtx.range("softmax"):
        max_value = tensor.max(dim=dim, keepdim=True).values
        shifted = tensor - max_value

        exp = torch.exp(shifted)
        denomi = torch.sum(exp, dim=dim, keepdim=True)

        return exp / denomi
