"""Character language-model architectures."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class CharRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        model_type: Literal["rnn", "gru", "lstm"] = "lstm",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_type not in {"rnn", "gru", "lstm"}:
            raise ValueError("model_type must be one of: rnn, gru, lstm")
        self.config = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "model_type": model_type,
            "dropout": dropout,
        }
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        recurrent_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[model_type]
        self.recurrent = recurrent_class(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor, hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None):
        embedded = self.embedding(tokens)
        activations, hidden = self.recurrent(embedded, hidden)
        return self.output(activations), hidden
