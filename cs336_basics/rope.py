import torch
from einops import rearrange
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1. 计算逆频率 inv_freq
        # 公式: theta_i = theta^(-2i / d_k), i 属于 [0, d_k/2)
        i = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / (theta ** (i / d_k))  # shape = (d_k // 2, )

        # 计算位置position
        # torch.arange(n) = range(n)
        m = torch.arange(max_seq_len).float()  # shape = (max_seq_len, )

        # 得到全量矩阵
        freqs_matrix = torch.outer(m, inv_freq)  # shape = (inv_freq, m)
        # shape 长成这样的目的是方便我们后续进行处理
        # results shape is (... seq_len, d_k // 2)
        # 所以为了方便起见，我们将我们的向量处理成为了shape = (inv_freqs, m)这样的维度

        self.register_buffer("cos_table", torch.cos(freqs_matrix), persistent=False)
        self.register_buffer("sin_table", torch.sin(freqs_matrix), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """arXiv:2104.09864v5 [cs.CL] 8 Nov 2023中的旋转矩阵高效实现方法实现，具体实现公式见论文"""
        # 1. 根据 token_positions 对预计算的 cos 和 sin 张量进行切片
        # 利用 PyTorch 的高级索引和广播机制：
        # cos 和 sin 的形状将被广播为 (..., seq_len, d_k // 2)
        # 这完美容忍了任意数量的批量维度（如 batch, heads 等）
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]
        # 这里的核心作用是调整我们的cos/sin_table的

        # 2. 使用 einops 将 x 最后一个维度拆分为 (d_k // 2, 2)
        # x_pairs 形状 (..., seq_len, d_k // 2, 2)
        x_pair = rearrange(x, "... (d r) -> ... d r", r=2)

        # 沿着最后一个维度进行解绑，根据 tensor.unbind() 的特性得到偶数索引和奇数索引的分量
        x_even, x_odd = x_pair.unbind(dim=-1)

        # 按照高效矩阵乘法实现公式得到
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        # X_2n+1 * cos(token) + X_2n+1 * sin(token)
        # X_2n+2 * cos(token) - X_2n+2 * sin(token)

        # 重新交错合并
        # 在将它们重叠回去
        stacked = torch.stack([x_even_rot, x_odd_rot], dim=-1)

        # 再使用 einops 展平最后两个维度，恢复(..., seq_len, d_k)
        out = rearrange(stacked, "... d r -> ... (d r)")

        return out


if __name__ == "__main__":
    # y = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    # i = torch.tensor([0, 1])
    # j = torch.tensor([1, 2])
    # print(y)
    # print(y[i, j])
    # print(y[i, :, j])
    # print(f"{y[0, :, 1]}\n{y[1, :, 2]}")
    # x = torch.range(1, 12).reshape(6, 2)
    # y, z = x.unbind(dim=-1)
    # t = torch.stack([y, z], dim=-1)
    # print(x)
    # print(y, z)
    # print(t)
    pass
