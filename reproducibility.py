from __future__ import annotations

"""Matrix Factorization model used as the traditional recommendation baseline.

The model represents users and items with latent vectors and scores a pair
using their dot product plus user, item and global bias terms.
"""

import torch
from torch import nn


class MatrixFactorizationModel(nn.Module):
    """
    Predict implicit preference from user/item latent factors and bias terms.
    
    The forward method returns logits so BCEWithLogitsLoss can apply the sigmoid
    transformation in a numerically stable way.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int,
    ) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(n_users, factors)
        self.item_embedding = nn.Embedding(n_items, factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate one logit for every user-item pair in the input batch."""
        user_vector = self.user_embedding(users)
        item_vector = self.item_embedding(items)

        interaction = torch.sum(user_vector * item_vector, dim=1)
        user_bias = self.user_bias(users).squeeze(-1)
        item_bias = self.item_bias(items).squeeze(-1)

        return interaction + user_bias + item_bias + self.global_bias
