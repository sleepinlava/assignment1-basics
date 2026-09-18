import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()

        self.d_model = d_model
        # modle hidden num of dims

        self.eps = float(eps)

        # We do not need write like: torch.tensor(eps)
        # because When `eps` is used as a Python float in CUDA tensor operations,
        #  there is absolutely no need to "move it to the GPU."
        # PyTorch handles such scalar constants directly as kernel arguments,
        #  without allocating dedicated GPU memory for them.
        # Epsilon value for numerical stability

        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(self.d_model))

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = (x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps).sqrt()
        return result.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.to(self.weight.device)
        return x / self.rms(x) * self.weight


if __name__ == "__main__":
    pass
