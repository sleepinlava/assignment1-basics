import torch
from einops import einsum
from torch import nn
from torch.nn.init import trunc_normal_


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        shape_tensor = torch.ones(out_features, in_features, dtype=dtype, device=device)
        rand_w_tensor = trunc_normal_(shape_tensor)
        # 使用einsum中最重要的内容就是搞清楚你的每一个维度到底是干什么的

        self.weight = nn.Parameter(rand_w_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = rearrange(x, "... input_shape -> input_shape ...")
        result = einsum(self.weight, x, "output_shape input_shape, ... input_shape -> ... output_shape")
        # result = rearrange(result, "input_shape ... -> ... input_shape")
        # einsum 使得我们只需要关注各个维度的实际意义即可
        # 一般的来说，我们的投入linear layer的tensor，最后一个维度一边会作为我们的input_dim
        # input : tensor -> [... input]
        # weight : tensor -> [input, output]
        # result_tensor -> [... output]
        # 但在这个任务中，我们不可以将weight进行转置，所以就需要我们先转置，然后乘法，最后再转置
        return result


if __name__ == "__main__":
    pass
