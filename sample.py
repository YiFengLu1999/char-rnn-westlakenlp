#!/usr/bin/env python3
"""Generate text with a PyTorch character language-model checkpoint."""

from __future__ import annotations

import argparse

import torch

from char_rnn.checkpoints import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to best.pt or last.pt")
    parser.add_argument("--prompt", default="", help="Optional text used to initialize the recurrent state")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="Restrict sampling to the k most likely characters; 0 disables it")
    parser.add_argument("--greedy", action="store_true", help="Always choose the most likely character")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, greedy: bool) -> int:
    if greedy:
        return int(logits.argmax(dim=-1).item())
    if temperature <= 0:
        raise ValueError("--temperature must be positive unless --greedy is used")
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())


def main() -> None:
    args = parse_args()
    if args.length < 1:
        raise ValueError("--length must be positive")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    model.eval()

    prompt = args.prompt
    if prompt:
        prompt_tokens = tokenizer.encode(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, hidden = model(prompt_tokens)
        current_logits = logits[:, -1, :]
    else:
        hidden = None
        current_logits = torch.zeros(1, tokenizer.vocab_size, device=device)

    generated: list[str] = [prompt]
    with torch.no_grad():
        for _ in range(args.length):
            token = sample_token(current_logits, args.temperature, args.top_k, args.greedy)
            generated.append(tokenizer.decode([token]))
            current = torch.tensor([[token]], device=device)
            logits, hidden = model(current, hidden)
            current_logits = logits[:, -1, :]
    print("".join(generated))


if __name__ == "__main__":
    main()
