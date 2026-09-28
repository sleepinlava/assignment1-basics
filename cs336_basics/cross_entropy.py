import torch

# from cs336_basics.softmax import softmax


def cross_entropy(logits: torch.Tensor, tagert: torch.Tensor) -> torch.Tensor:
    """some function MAY BE DESTORYED YOUR TENSOR SHAPE 😅
    And use your FUCKING REAL NAME 😅😅😅,
    DO NOT Because use the FUCKING STUPID name to disturb other one's brain
    """
    # max不声明keepdim，会导致max隐式的转变你的维度 * 2
    # 在后续的广播机制时出现隐藏的错误
    # 所以在用torch中的操作是，如果有keepdim这个参数时
    # 要更加关心这个的存在，是否会隐式的导致我们后续梯度计算出现隐藏错误
    # 要从每一个维度上进行取最大值，不然从数学上层面没有实际应用，回去看我们的数学公式

    dtype = logits.dtype
    logits = logits.to(torch.float32)

    max = logits.max(dim=-1, keepdim=True).values

    stable_log = torch.log(torch.exp(logits - max).sum(dim=-1))

    length_target = torch.arange(logits.size(-2))

    result = (max.squeeze(-1) + stable_log - logits[..., length_target, tagert]).mean()
    # 为什么这里需要删掉最后一个维度呢？
    # 由于我们max的shape=(batch, seq_len, 1) 数学上的形状符合logits
    # 但是后面我们求和一系列的操作，使得logits塌缩成为(batch, seq_len)这个维度
    # 如果不挤掉，就会导致触发广播机制
    # 广播机制真够是味道的巧克力吧 😅

    return result.to(dtype)


if __name__ == "__main__":
    pass
    # pass
    # a = torch.rand([4,4,4])
    # a = torch.arange(8).reshape(2, 2, 2)
    # print(a)
    # print(a.sum(dim=-1, keepdim=True))
    # print(a.sum())
    # print(a)
    # # print(a[...,-1])
    # print(a.max(dim=-1).values,'\n', a.max(dim=-1).indices)
    # print(a.max(dim=-1, keepdim = True).values,'\n', a.max(dim=-1).indices)
    # print(a[...,-1])
    # a = torch.arange(8).reshape(2,4)
    # # b = torch.arange(0, 8, 4).unsqueeze(-1)
    # b = torch.arange(2).reshape(2,1)
    # print(a)
    # print(b)
    # # print(a[b])
    # print(torch.gather(a,-2,b))
    # dim 指定在 input 的哪个维度上查索引；
    # index 是一个与输出同形状的张量，它的每个元素给出沿 dim 维度要取的索引值；
    # 其他维度的坐标由 index 中该元素的位置决定；
    # index 中的值必须在 [0, input.size(dim)) 范围内。
    #
    # a = torch.ones(3, 3)
    # print(a, a.sum(dim=-1, keepdim=True), a.sum(dim=-1, keepdim=True).mean())

    # a = torch.rand(2, 2, 2)
    # b = torch.rand(1,2)
    # c = torch.rand(1, 2)
    # print(f'{a} \n {b} \n {c}')
    # print(f'{a + b + c}')
    # max 需要挤掉的原因， 广播发力了
#
#     print(torch.arange(8).reshape(2, 4))
#     print(torch.arange(8).reshape(2, 4).size(-1))
