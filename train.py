#!/usr/bin/env python3
"""Train a Unicode-safe character-level RNN, GRU, or LSTM."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from char_rnn.checkpoints import save_checkpoint
from char_rnn.data import CharacterTokenizer, TextBatches
from char_rnn.model import CharRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tinyshakespeare/input.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model", choices=("rnn", "gru", "lstm"), default="lstm")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
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


def evaluate(model: CharRNN, batches, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in batches:
            logits, _ = model(inputs.to(device))
            total_loss += criterion(logits.reshape(-1, logits.size(-1)), targets.to(device).reshape(-1)).item()
    return total_loss / len(batches)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    batches = TextBatches.from_file(args.data, tokenizer, args.batch_size, args.seq_length, args.train_fraction)
    model = CharRNN(tokenizer.vocab_size, args.embedding_dim, args.hidden_size, args.num_layers, args.model, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    print(f"device={device}; characters={len(text):,}; vocabulary={tokenizer.vocab_size}; "
          f"train_batches={batches.train_count}; val_batches={batches.val_count}; "
          f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    best_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in batches.split("train"):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs.to(device))
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), targets.to(device).reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= batches.train_count
        val_loss = evaluate(model, batches.split("val"), criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        metadata = {"epoch": epoch, "history": history, "train_args": vars(args)}
        save_checkpoint(args.output_dir / "last.pt", model, tokenizer, **metadata)
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(args.output_dir / "best.pt", model, tokenizer, **metadata)
            print(f"  saved new best checkpoint (val_loss={best_loss:.4f})")


if __name__ == "__main__":
    main()
