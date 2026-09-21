import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_tensor = x.max(dim=dim, keepdim=True).values
    # max不声明keepdim，会导致max隐式的转变你的维度
    # 在后续的广播机制时出现隐藏的错误
    # 所以在用torch中的操作是，如果有keepdim这个参数时
    # 要更加关心这个的存在，是否会隐式的导致我们后续梯度计算出现隐藏错误

    # softmax 在某个指定维度上把原始分数归一化成概率分布；
    # 这个概率分布可以作为后续加权组合的权重；
    # 后续向量的方向由这些权重和其他向量加权求和决定；
    # 最大概率项只是权重最大的项，通常不是唯一决定因素。
    up = torch.exp(x - max_tensor)
    down = torch.sum(torch.exp(x - max_tensor), dim=dim, keepdim=True)
    # 我们可以利用这一性质实现数值稳定性
    # 通常，我们会从 𝑣 的所有元素中减去 𝑣 的最大元素
    # 使新的最大元素为 0。
    # softmax 数学公式中，下面为沿着dim维度进行求和
    return up / down


if __name__ == "__main__":
    pass
# class softmax(nn.Module):
#     def __init__(
#         self, x: torch.Tensor, dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None
#     ) -> None:
#         super().__init__()
#         self.x = x
#         self.dim = dim
#         self.device = device
#         self.dtype = dtype

#     def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
#         max_tensor = x.max()
#         return torch.exp(x[dim] - max_tensor) / torch.sum(torch.exp(x))
