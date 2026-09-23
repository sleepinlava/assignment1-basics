import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn import init

from cs336_basics import rope, softmax


def scaled_dot_product_attention(
    keys: torch.Tensor, queries: torch.Tensor, values: torch.Tensor, bool_mask: torch.Tensor
) -> torch.Tensor:
    """keys, queries -> (batch_size, ..., seq_len, d_K), values -> (batch_size, ..., seq_len, d_v)

    The original transformer time complex -> O(n^2)

    shape (batch_size, ..., seq_len, d_K) -> (batch_size, ..., seq_len, seq_len) -> (batch_size, ..., seq_len, d_v)

    bool_mask -> (..., seq_len_queries, seq_len_keys)
    """
    # keys_t = rearrange(keys, "batch ... seq_len d_k -> batch ... d_k seq_len")
    scale_fact = keys.shape[-1] ** (0.5)
    QK_multi = einsum(
        queries,
        keys,
        "batch ... seq_len_queries d_k, batch ...  seq_len_keys d_k -> batch ... seq_len_keys seq_len_queries",
        # 这里为什么需要反常规来处理？ 注意看一下题目中对于Q K V维度的描述
    )
    # notice that use different var name
    if bool_mask is not None:
        QK_results = (QK_multi / scale_fact).masked_fill(~bool_mask, -1e8)
        # ~ 这里的作用是取反，具体的例子可以见反面的main的demo
    return softmax.softmax((QK_results), dim=-1) @ values


# 为了增加学生对于维度的理解，专门给我把结果反过来处理吗
# "按惯例(可能令人有些困惑)" 专门要写成 不是常理的 K^T * Q 😅
# 哈基Percy,你赢了😅


class multihead_self_attention_class(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """X_input.shape(batch, seq_len, d_embedding)
        -------------
        -> W_K.shape(h * d_K, d_embedding) # note: d_embedding = heads * d_embedding
        -> W_Q.shape(h * d_Q, d_embedding)
        -> W_V.shape(h * d_V, d_embedding)
        -> W_O.shape(h * d_v, d_embedding)
        --------------
        K.shape(batch, seq_len, d_embedding, h * d_K) -> (batch, heads, seq_len, d_K)
        Q.shape(batch, heads, seq_len, d_Q)
        V.shape(batch, heads, seq_len, d_V)
        -------------
        QK^T.shape(batch, heads, seq_len, seq_len)
        Softmax(QK^T/scalce_fact) @ V .shape(batch, heads, seq_len, d_V) -> compact,dim=heads ->(batch, seq_lens, d_embedding)
        X_output = X_input * W_O .shape(batch, seq_lens, d_embedding)
        we find that input.shape SHOULD equal to output.shape
        """

        # d_model <-> embedding_model
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_qkv = int(d_model / num_heads)
        self.scale_fact = (d_model / num_heads) ** 0.5
        self.device = device
        self.dtype = dtype

        # self.W_k = nn.Parameter(kaiming_normal_(torch.Tensor(self.d_model, self.d_model)))

        weights_kernel = init.kaiming_normal_(torch.Tensor(4, self.d_model, self.d_model))

        self.W_k = nn.Parameter(weights_kernel[0, :])
        self.W_q = nn.Parameter(weights_kernel[1, :])
        self.W_v = nn.Parameter(weights_kernel[2, :])
        self.W_o = nn.Parameter(weights_kernel[3, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # **attention <-> d_k = d_q**
        #
        # 矩阵乘法要转置思密达
        # 为什么能够侥幸通过形状检查思密达
        # 因为这是个正方矩阵思密达
        # 怎么搞都正确从而隐藏了错误思密达
        # 😅😅😅
        #
        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T

        Q = rearrange(q, "batch seq_len_q (h d_q) -> batch h seq_len_q d_q", h=self.num_heads)
        K = rearrange(k, "batch seq_len_k (h d_k) -> batch h seq_len_k d_k", h=self.num_heads)
        V = rearrange(v, "batch seq_len_v (h d_v) -> batch h seq_len_v d_v", h=self.num_heads)

        QK_result = (
            einsum(Q, K, "batch head seq_len_q d, batch head seq_len_k d -> batch head seq_len_q seq_len_k")
            / self.scale_fact
        )

        bool_mask = torch.triu(torch.ones(QK_result.shape[-2], QK_result.shape[-1]), 1).bool()

        QK_masked = QK_result.masked_fill(bool_mask, -1e8)

        socres = softmax.softmax(QK_masked, dim=-1) @ V

        # socres = softmax.softmax(QK_result, dim=-1) @ V

        # we should follow the dim is -2?, because Q<->K not K <-> Q
        # No, think more we found that the meaning is right

        O = rearrange(socres, "batch h seq_len d_v -> batch seq_len (h d_v)", h=self.num_heads)

        # return softmax(QK_result, dim=-1), softmax(QK_masked, dim=-1)
        return O @ self.W_o.T


class multihead_self_attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_sequnece_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """X_input.shape(batch, seq_len, d_embedding)
        -------------
        -> W_K.shape(h * d_K, d_embedding) # note: d_embedding = heads * d_embedding
        -> W_Q.shape(h * d_Q, d_embedding)
        -> W_V.shape(h * d_V, d_embedding)
        -> W_O.shape(h * d_v, d_embedding)
        --------------
        K.shape(batch, seq_len, d_embedding, h * d_K) -> (batch, heads, seq_len, d_K)
        Q.shape(batch, heads, seq_len, d_Q)
        V.shape(batch, heads, seq_len, d_V)
        -------------
        QK^T.shape(batch, heads, seq_len, seq_len)
        Softmax(QK^T/scalce_fact) @ V .shape(batch, heads, seq_len, d_V) -> compact,dim=heads ->(batch, seq_lens, d_embedding)
        X_output = X_input * W_O .shape(batch, seq_lens, d_embedding)
        we find that input.shape SHOULD equal to output.shape
        """

        # d_model <-> embedding_model
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_qkv = int(d_model / num_heads)
        self.scale_fact = (d_model / num_heads) ** 0.5
        self.theta = theta
        self.max_sequnece_len = max_sequnece_len
        self.device = device
        self.dtype = dtype

        weights_kernel = init.kaiming_normal_(torch.Tensor(4, self.d_model, self.d_model))

        self.W_k = nn.Parameter(weights_kernel[0, :])
        self.W_q = nn.Parameter(weights_kernel[1, :])
        self.W_v = nn.Parameter(weights_kernel[2, :])
        self.W_o = nn.Parameter(weights_kernel[3, :])

    def forward(
        self,
        x: torch.Tensor,
        token_position: torch.Tensor,
    ) -> torch.Tensor:
        # **attention <-> d_k = d_q**

        # 矩阵乘法要转置思密达
        # 为什么能够侥幸通过形状检查思密达
        # 因为这是个正方矩阵思密达
        # 怎么搞都正确从而隐藏了错误思密达
        # 😅😅😅
        q = x @ self.W_q.T
        k = x @ self.W_k.T
        v = x @ self.W_v.T

        Rope = rope.RotaryPositionalEmbedding(self.theta, self.d_qkv, self.max_sequnece_len)

        # we lack of the causal mask and RoPE

        Q = rearrange(q, "batch seq_len_q (h d_q) -> batch h seq_len_q d_q", h=self.num_heads)
        K = rearrange(k, "batch seq_len_k (h d_k) -> batch h seq_len_k d_k", h=self.num_heads)
        V = rearrange(v, "batch seq_len_v (h d_v) -> batch h seq_len_v d_v", h=self.num_heads)

        Q_r = Rope.forward(Q, token_position)
        K_r = Rope.forward(K, token_position)

        QK_result = (
            einsum(Q_r, K_r, "batch head seq_len_q d, batch head seq_len_k d -> batch head seq_len_q seq_len_k")
            / self.scale_fact
        )

        bool_mask = torch.triu(torch.ones(QK_result.shape[-2], QK_result.shape[-1]), 1).bool()

        QK_masked = QK_result.masked_fill(bool_mask, -1e8)

        socres = softmax.softmax(QK_masked, dim=-1) @ V

        # socres = softmax.softmax(QK_result, dim=-1) @ V

        # we should follow the dim is -2?, because Q<->K not K <-> Q
        # No, after thinking more we found that the meaning is right

        O = rearrange(socres, "batch h seq_len d_v -> batch seq_len (h d_v)", h=self.num_heads)

        # return softmax(QK_result, dim=-1), softmax(QK_masked, dim=-1)

        return O @ self.W_o.T


if __name__ == "__main__":
    pass

    # print(init.kaiming_normal_(torch.ones(4,4,4)))

    # pass
    # a = torch.arange(27).reshape(3,3,3)
    # print(a)
    # print(a.T)
    # pass
    # a = torch.arange(32).reshape(2, 4, 4)
    # print(a)
    # print(a.max())
    # print(a[-2,...].unsqueeze(-1))
    # pass
    # x = torch.ones(5, dtype=torch.bool)
    # print(~x)
    # x : int = 4
    # y : int = 2
    # print(int(x / y))
    # a = torch.arange(12).reshape(2, 3, 2)
    # b = torch.rand(12).reshape(2, 3, 2)
    # print(f"{a}\n{b}")
    # print(torch.concat((a, b), dim=-1))
    # a = torch.arange(64).reshape(2, 2, 4, 4)
    # b = torch.triu(a, 1).to(dtype=torch.bool)  # 构造一个01下三角矩阵，用来作为bool_mask
    # c = torch.triu(a, 0).to(dtype=torch.bool)
    # mask2d = torch.triu(torch.ones(4, 4), 1).bool()
    # print(mask2d)
    # # c = torch.tril(a, 0).to(dtype=torch.bool)  # 构造了一个01上三角矩阵，用来我们bool_mask一般作用于最后两个维度
    # print(a)
    # print(a.masked_fill(mask2d,99))
    # print(a.masked_fill(b,99))
    # print(torch.equal(a.masked_fill(mask2d,99), a.masked_fill(b,99)))
    # print(b)
    # print(c)
    # print(a.masked_fill(b, 99))
    # print(a.masked_fill(c, 99))
    # print(kaiming_normal_(torch.Tensor(4, 4)))
    # print(a[3])
    # print(a.select(-1,1))
    # batch = 2; t = 4; d = 8; h = 2
    # x = torch.rand(2, 4, 8)
    # model = multihead_self_attention(8, 2)
    # # print(f"{model.W_k.shape}\n{model.W_q.shape}\n{model.W_v.shape}\n{model.W_o.shape}\n this is split line")
    # # t1, t2 = model.MHA(x, 1000, 1024)
    # print(model.MHA(x, 1000, 1024))
    # print(f"here is unmasked{t1},\n here is masked{t2}")
    # print(x)
    # a = torch.arange(32).reshape(2, 2, 2, 2, 2)
    # print(a.shape[-1], a.shape[-2])
