import torch

from cs336_basics.bpe_train import BPEModel
from cs336_basics.transformer import transformer_lm

tokenizer = BPEModel.load("/home/sleepinlava/Projects/python/cs336/assignment1-basics/checkpoints/bpe_tinystory.pkl")

model = transformer_lm(9999, 1024, 12, 768, 12, 2048, 0.01)
model.load_state_dict(torch.load("test.pth", map_location="cpu", weights_only=True))


@torch.inference_mode()
def generate(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    context_length=256,
    eos_token_id=None,
):
    for _ in range(max_new_tokens):
        # 只将模型支持的上下文送入 Transformer
        context = input_ids[:, -context_length:]

        # (B, L) -> (B, L, V)
        logits = model(context)

        # 只需要最后一个位置的 logits
        # (B, L, V) -> (B, V)
        next_token_logits = logits[:, -1, :]

        # Temperature scaling
        next_token_logits = next_token_logits / temperature

        # Top-k filtering
        k = min(top_k, next_token_logits.size(-1))

        values, _ = torch.topk(next_token_logits, k=k, dim=-1)

        threshold = values[:, -1, None]

        next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold, float("-inf"))

        # Softmax: logits -> probability
        probs = torch.softmax(next_token_logits, dim=-1)

        # 从概率分布中采样
        next_token = torch.multinomial(probs, num_samples=1)

        # 拼接到已有输入
        input_ids = torch.cat([input_ids, next_token], dim=1)

        token_id = next_token.item()

        # 交给外部逐 token 处理
        yield token_id

        if eos_token_id is not None and token_id == eos_token_id:
            break


def stream_generate(model, tokenizer, prompt):
    # Prompt -> Token IDs
    input_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor([input_ids], dtype=torch.long)

    generated_ids = []
    printed_text = ""

    print(prompt, end="", flush=True)

    for token_id in generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        context_length=256,
    ):
        generated_ids.append(token_id)

        # 解码目前生成的完整 token 序列
        current_text = tokenizer.decode(generated_ids)

        # 打印新增部分
        if current_text.startswith(printed_text):
            new_text = current_text[len(printed_text) :]
            print(new_text, end="", flush=True)

        else:
            # 遇到不完整 UTF-8 序列等情况时，
            # 不直接打印可能错误的增量
            new_text = ""

        if current_text.startswith(printed_text):
            printed_text = current_text

    print()


def main():
    while True:
        prompt = input("\nUser > ")

        if prompt.strip().lower() in ["exit", "quit"]:
            break

        print("Model > ", end="", flush=True)

        stream_generate(model=model, tokenizer=tokenizer, prompt=prompt)


if __name__ == "__main__":
    main()
