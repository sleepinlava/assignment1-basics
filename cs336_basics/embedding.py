import torch
from torch import nn
from torch.nn.init import trunc_normal_


class embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        embedding_shape_matrix = torch.ones(self.num_embeddings, self.embedding_dim)
        # in my opinion,
        # embedding_dim <-> batch_num | num_embeddings <-> seq_num
        embedding_matrix = trunc_normal_(embedding_shape_matrix, a=-1, b=1)

        self.weight = nn.Parameter(embedding_matrix)
        # embedding martix size is (num_embedding * embedding_dim)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:

        # 在 PyTorch 里，用一个整数张量去索引另一个张量，
        # 叫做“高级索引（advanced indexing）”。它的规则是：
        # 把索引张量里的每个值，都当成被索引张量第 0 维的下标，
        # 逐个取出对应的行，然后用索引张量的形状作为输出的形状。

        # token_ids -> (b, l) ∈ [batch_size, Seq_length]
        # Seq_length -> [token_id_1, token_id2, token_id3, ...]
        # l <-> token_id_i

        # here we can not use tolist(), because it will significantly slow down our perfermance in HPC
        # flow
        # : GPU -> RAM -> CPU -> process :(

        # what is torch.LongTensor?
        # it is a 64 bits intgeters -> troch.int64
        # always in CPU(and can put it into GPU), and not support to grad
        # we usually use it in index, labels, int computes
        
        # token_ids 是形状为 (B, L) 的整数矩阵，每个元素是词表中的 token id，取值范围 [0, V-1]。
        # Embedding 权重 weight 形状为 (V, D)，其中 V 是词表大小，D 是 embedding 维度。
        # 高级索引 self.weight[token_ids] 的逻辑是：对每个 token_ids[b, l] 取 weight 的第
        # token_ids[b, l] 行，得到一个 D 维向量，再按原 (B, L) 形状堆叠，输出 (B, L, D)。
        # B 和 L 只决定取多少次和结果如何排布，不决定取哪一行；取哪一行完全由 token_id 决定。
        # 分词器与 embedding 必须共享同一张词表，才能保证 id 与行号对应。
        # 底层由 C++/CUDA 并行实现，并非 Python 循环，因此效率极高。
        # nn.Embedding 的 forward 本质上就是 self.weight[token_ids]
        return self.weight[token_ids]


if __name__ == "__main__":
    # pass
    model = embedding(10, 10)
    print(model.weight)
    token_id = torch.LongTensor([5, 4])
    print(model.forward(token_id))
