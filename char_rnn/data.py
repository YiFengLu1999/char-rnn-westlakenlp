"""Unicode-safe character tokenization and fixed-length language-model batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch


@dataclass(frozen=True)
class CharacterTokenizer:
    """A deterministic character vocabulary based on Python Unicode characters."""

    characters: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Training text is empty.")
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    @property
    def stoi(self) -> dict[str, int]:
        return {character: index for index, character in enumerate(self.characters)}

    def encode(self, text: str) -> torch.Tensor:
        mapping = self.stoi
        unknown = sorted(set(text) - set(mapping))
        if unknown:
            shown = "".join(unknown[:10])
            raise ValueError(f"Text contains {len(unknown)} character(s) absent from the vocabulary: {shown!r}")
        return torch.tensor([mapping[character] for character in text], dtype=torch.long)

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().tolist()
        return "".join(self.characters[index] for index in tokens)

    def state_dict(self) -> dict[str, list[str]]:
        return {"characters": list(self.characters)}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[str]]) -> "CharacterTokenizer":
        return cls(tuple(state["characters"]))


class TextBatches:
    """Prepares non-overlapping BPTT batches from one text corpus.

    Each item contains ``(inputs, targets)`` with shape ``[batch_size, seq_length]``.
    Python strings are iterated as Unicode code points, so Chinese characters remain intact.
    """

    def __init__(self, tokens: torch.Tensor, batch_size: int, seq_length: int, train_fraction: float) -> None:
        if batch_size < 1 or seq_length < 1:
            raise ValueError("batch_size and seq_length must be positive.")
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1.")

        usable = ((tokens.numel() - 1) // (batch_size * seq_length)) * batch_size * seq_length
        if usable == 0:
            raise ValueError("Text is too short for this batch_size and seq_length.")

        inputs = tokens[:usable].view(batch_size, -1)
        targets = tokens[1 : usable + 1].view(batch_size, -1)
        self._batches = [
            (inputs[:, start : start + seq_length], targets[:, start : start + seq_length])
            for start in range(0, inputs.size(1), seq_length)
        ]
        if len(self._batches) < 2:
            raise ValueError("Text is too short to create both training and validation batches.")
        self.train_count = int(len(self._batches) * train_fraction)
        self.train_count = max(1, min(self.train_count, len(self._batches) - 1))
        self.val_count = len(self._batches) - self.train_count

    @classmethod
    def from_file(
        cls, path: str | Path, tokenizer: CharacterTokenizer, batch_size: int, seq_length: int, train_fraction: float
    ) -> "TextBatches":
        return cls(tokenizer.encode(Path(path).read_text(encoding="utf-8")), batch_size, seq_length, train_fraction)

    def split(self, name: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if name == "train":
            return self._batches[: self.train_count]
        if name == "val":
            return self._batches[self.train_count :]
        raise ValueError("Split must be 'train' or 'val'.")
