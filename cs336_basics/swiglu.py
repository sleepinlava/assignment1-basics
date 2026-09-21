import torch
from torch import nn, sigmoid

from cs336_basics import linear


class positionwise_feedward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.device | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.weight_1 = linear.Linear(d_model, d_ff)
        self.weight_2 = linear.Linear(d_ff, d_model)
        self.weight_3 = linear.Linear(d_model, d_ff)

        """

        # self.weight_1 = Linear(d_ff, d_model).weight
        # self.weight_2 = Linear(d_model, d_ff).weight
        # self.weight_3 = Linear(d_ff, d_model).weight

        （标准写法）：self.w1 = nn.Linear(d_model, d_ff, bias=False)
        在前向传播时，你只需要写 x = self.w1(x)。nn.Linear 内部自动帮你完成了矩阵乘法
        （提取权重写法）：self.weight_1 = Linear(d_ff, d_model).weight
        你只拿到了一个裸的 Tensor（准确说是 nn.Parameter）。
        在前向传播时，你必须放弃层的调用方式，改为手动调用底层的函数：
        x = F.linear(x, self.weight_1)。这大大降低了代码的可读性。

        参数注册与设备管理（Framework Magic）
        PyTorch 的 nn.Module 有一个非常强大的机制：属性赋值自动注册。
        当你写 self.w1 = nn.Linear(...) 时，PyTorch 会自动识别这是一个子模块，
        并将其内部的 weight 和 bias（如果有）注册到整个模型的参数列表中。
        当你写 self.weight_1 = Linear(...).weight 时，你实际上是创建了一个临时 Linear 对象，
        提取了它的权重，然后丢弃了那个对象。虽然 nn.Parameter 赋值给 nn.Module 属性时也会被注册
        （这是 PyTorch 的特性），但这会导致：
        初始化逻辑被破坏（原本应该在 __init__ 中统一初始化的逻辑，现在散落了）。
        当你调用 model.to('cuda') 时，虽然参数也会跟着走，
        但如果结构复杂（比如涉及多个分支），手动管理极容易出错。

        """

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        """x/(1 + exp(x))"""
        return x * sigmoid(x)

    def swiglu(self, x: torch.Tensor):
        # swiglu = new FFN
        # here we combine the gate-control with activate function
        return self.weight_2.forward(self.silu(self.weight_1.forward(x)) * self.weight_3(x))


if __name__ == "__main__":
    pass
