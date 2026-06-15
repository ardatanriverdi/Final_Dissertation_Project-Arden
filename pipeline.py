from __future__ import annotations

"""Shared PyTorch data structures used by both recommender models.

Keeping the same Dataset implementation for MF and NCF helps ensure that
both models receive the same training examples.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    Convert a binary interaction DataFrame into tensors consumed by a DataLoader.
    
    Required columns are user_idx, item_idx and label.
    """
    def __init__(self, interactions: pd.DataFrame) -> None:
        required = {"user_idx", "item_idx", "label"}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(f"Missing training columns: {sorted(missing)}")

        self.users = torch.tensor(
            interactions["user_idx"].to_numpy(),
            dtype=torch.long,
        )
        self.items = torch.tensor(
            interactions["item_idx"].to_numpy(),
            dtype=torch.long,
        )
        self.labels = torch.tensor(
            interactions["label"].to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Return the number of user-item-label training examples."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Return one user index, item index and binary label."""
        return (
            self.users[index],
            self.items[index],
            self.labels[index],
        )
