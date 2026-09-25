import torch
from torch import nn
from torch.nn.modules import transformer

from cs336_basics.embedding import embedding
from cs336_basics.linear import Linear
from cs336_basics.pre_norm_transformer import pre_norm_transformer
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.softmax import softmax


class transformer_lm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta

        self.embedding_block = embedding(vocab_size, d_model)
        self.pre_norm_transformer_block = nn.ModuleList(
            [pre_norm_transformer(d_model, num_heads, d_ff, context_length, theta) for _ in range(self.num_layers)]
        )
        self.norm_block = RMSNorm(d_model)
        self.mlp_block = Linear(d_model, vocab_size)

    def forward(self, input_tesor) -> torch.Tensor:
        transformer_tensor = self.embedding_block.forward(input_tesor)
        for layer in self.pre_norm_transformer_block:
            transformer_tensor = layer(transformer_tensor)
        transformer_tensor = self.norm_block.forward(transformer_tensor)
        transformer_tensor = self.mlp_block.forward(transformer_tensor)
        # transformer_tensor = softmax(transformer_tensor, dim=-1)
        return transformer_tensor


if __name__ == "__main__":
    pass
