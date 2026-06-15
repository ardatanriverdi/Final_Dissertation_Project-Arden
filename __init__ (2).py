from __future__ import annotations

"""Neural Collaborative Filtering model used as the neural benchmark.

User and item embeddings are concatenated and passed through a multilayer
perceptron to learn non-linear interaction patterns.
"""

from collections.abc import Sequence

import torch
from torch import nn


class NeuralCollaborativeFilteringModel(nn.Module):
    """
    Learn non-linear user-item interactions through embeddings and an MLP.
    
    The final layer produces a single logit for binary implicit-feedback training.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_layers: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate one logit for every user-item pair in the input batch."""
        user_vector = self.user_embedding(users)
        item_vector = self.item_embedding(items)
        combined = torch.cat([user_vector, item_vector], dim=-1)

        return self.mlp(combined).squeeze(-1)
