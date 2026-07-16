"""Checkpoint serialization helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from .data import CharacterTokenizer
from .model import CharRNN


def save_checkpoint(path: str | Path, model: CharRNN, tokenizer: CharacterTokenizer, **metadata) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_config": model.config, "model_state": model.state_dict(), "tokenizer": tokenizer.state_dict(), **metadata},
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CharRNN, CharacterTokenizer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CharRNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, CharacterTokenizer.from_state_dict(checkpoint["tokenizer"]), checkpoint
