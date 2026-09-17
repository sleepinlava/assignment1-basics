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
        self.device = device
        self.dtype = dtype

        embedding_shape_matrix = torch.ones(
            self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device
        )
        # in my opinion,
        # embedding_dim <-> batch_num | num_embeddings <-> seq_num
        embedding_matrix = trunc_normal_(embedding_shape_matrix, a=-1, b=1).to(device)

        self.weight = nn.Parameter(embedding_matrix)
        # embedding martix size is (num_embedding * embedding_dim)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:

        # 在 PyTorch 里，用一个整数张量去索引另一个张量，
        # 叫做“高级索引（advanced indexing）”。它的规则是：
        # 把索引张量里的每个值，都当成被索引张量第 0 维的下标，
        # 逐个取出对应的行，然后用索引张量的形状作为输出的形状。

        # token_ids -> (b, l) ∈ [batch_size, Seq_length]
        # l is the position in the specific batch seq

        # here we can not use tolist(), because it will significantly slow down our perfermance in HPC
        # flow
        # : GPU -> RAM -> CPU -> process :(

        # what is torch.LongTensor?
        # it is a 64 bits intgeters -> troch.int64
        # always in CPU(and can put it into GPU), and not support to grad
        # we usually use it in index, labels, int computes
        return self.weight[token_ids]


if __name__ == "__main__":
    pass

