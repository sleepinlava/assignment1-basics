import torch
from einops import einsum, rearrange
from torch import nn

from cs336_basics import softmax


def scaled_dot_product_attention(
    keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, bool_mask: torch.Tensor
):
    """keys, queries -> (batch_size, ..., seq_len, d_K), values -> (batch_size, ..., seq_len, d_v)

    The original transformer time complex -> O(n^2)

    shape (batch_size, ..., seq_len, d_K) -> (batch_size, ..., seq_len, seq_len) -> (batch_size, ..., seq_len, d_v)

    bool_mask -> (..., seq_len_queries, seq_len_keys)
    """
    keys_t = rearrange(keys, "batch ... seq_len d_k -> batch ... d_k seq_len")
    scale_fact = keys.shape[-1] ** (0.5)
    QK_multi = einsum(
        queries,
        keys_t,
        "batch ... seq_len_queries d_k, batch ... d_k seq_len_keys_t -> batch ... seq_len_keys_t seq_len_queries",
        # 这里为什么需要反常规来处理？ 注意看一下题目中对于Q K V维度的描述
    )
    # notice that use different var name
    if bool_mask is not None:
        QK_results = (QK_multi / scale_fact).masked_fill(~bool_mask, -1e7)
        # ~ 这里的作用是取反，具体的例子可以见反面的main的demo
    return softmax.softmax((QK_results), dim=-1) @ values


# 为了增加学生对于维度的理解，专门给我把结果反过来处理吗
# "按惯例(可能令人有些困惑)" 专门要写成 不是常理的 K^T * Q 😅
# 哈基Percy,你赢了😅


class multihead_self_attention(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    # x = torch.ones(5, dtype=torch.bool)
    # print(~x)
    pass
