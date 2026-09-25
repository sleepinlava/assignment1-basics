import torch
from torch import nn

from cs336_basics import attention, rmsnorm, swiglu


class pre_norm_transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, theta: float) -> None:

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.context_length = context_length
        self.theta = theta

        self.norm_block_1 = rmsnorm.RMSNorm(d_model)
        self.norm_block_2 = rmsnorm.RMSNorm(d_model)
        self.attention_block = attention.multihead_self_attention(
            self.d_model, self.num_heads, self.theta, context_length
        )
        self.swiglu_block = swiglu.positionwise_feedward(d_model, d_ff)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        token_position = torch.arange(input_feature.shape[-2])
        tensor_0 = self.norm_block_1.forward(input_feature)
        tensor_1 = self.attention_block.forward(tensor_0, token_position)
        tensor_2 = tensor_1 + input_feature
        tensor_3 = self.norm_block_2.forward(tensor_2)
        tensor_4 = self.swiglu_block.swiglu(tensor_3)
        out_tensor = tensor_2 + tensor_4
        return out_tensor
