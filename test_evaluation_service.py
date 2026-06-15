from __future__ import annotations

"""Full-catalogue top-K evaluation service.

For each test user, all unseen items are scored and ranked. The resulting
recommendation list is evaluated with HR@K, NDCG@K and Recall@K.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from recommender_benchmark.services.data_service import (
    build_seen_items,
)


@dataclass(frozen=True)
class RankingResult:
    """Store ranking metrics and the number of evaluated users."""
    hr: float
    ndcg: float
    recall: float
    evaluated_users: int


def hit_rate_at_k(
    recommended_items: list[int],
    true_item: int,
    k: int,
) -> float:
    """Return 1 when the held-out item appears in the first K recommendations."""
    return float(true_item in recommended_items[:k])


def ndcg_at_k(
    recommended_items: list[int],
    true_item: int,
    k: int,
) -> float:
    """Reward a hit more strongly when it appears near the top of the ranked list."""
    top_k = recommended_items[:k]

    if true_item not in top_k:
        return 0.0

    rank = top_k.index(true_item) + 1
    return float(1.0 / np.log2(rank + 1))


def recall_at_k(
    recommended_items: list[int],
    true_items: set[int],
    k: int,
) -> float:
    """Return the proportion of relevant items retrieved in the top-K list."""
    if not true_items:
        return 0.0

    hits = len(set(recommended_items[:k]).intersection(true_items))
    return float(hits / len(true_items))


def recommend_top_k(
    model: nn.Module,
    *,
    user_idx: int,
    seen_items: set[int],
    n_items: int,
    k: int,
    device: torch.device,
    candidate_batch_size: int,
) -> list[int]:
    """
    Score every unseen catalogue item for one user and return the highest K.
    
    Candidate scoring is batched to control memory use, while argpartition avoids
    fully sorting the complete catalogue when only K items are required.
    """
    # Remove observed training items from the recommendation candidate set.
    candidates = np.array(
        [
            item_idx
            for item_idx in range(n_items)
            if item_idx not in seen_items
        ],
        dtype=np.int64,
    )

    if candidates.size == 0:
        return []

    scores: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(candidates), candidate_batch_size):
            batch = candidates[start : start + candidate_batch_size]

            users = torch.full(
                (len(batch),),
                user_idx,
                dtype=torch.long,
                device=device,
            )
            items = torch.tensor(
                batch,
                dtype=torch.long,
                device=device,
            )

            logits = model(users, items)
            scores.append(torch.sigmoid(logits).cpu().numpy())

    combined_scores = np.concatenate(scores)
    top_count = min(k, len(candidates))

    # Select top candidates in linear time before ordering only those K items.
    top_indices = np.argpartition(
        combined_scores,
        -top_count,
    )[-top_count:]

    ordered_indices = top_indices[
        np.argsort(combined_scores[top_indices])[::-1]
    ]

    return candidates[ordered_indices].astype(int).tolist()


def evaluate_ranking(
    model: nn.Module,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
    k: int,
    device: torch.device,
    candidate_batch_size: int,
    max_eval_users: int | None,
    seed: int,
) -> RankingResult:
    """
    Evaluate a model using the same full-catalogue protocol for every test user.
    
    Seen training items are excluded, but the held-out test item remains eligible.
    """
    seen_by_user = build_seen_items(train)
    users = test["user_idx"].drop_duplicates().astype(int).to_numpy()

    if max_eval_users is not None and len(users) > max_eval_users:
        rng = np.random.default_rng(seed)
        users = rng.choice(
            users,
            size=max_eval_users,
            replace=False,
        )

    selected = test[test["user_idx"].isin(users)]

    hr_values: list[float] = []
    ndcg_values: list[float] = []
    recall_values: list[float] = []

    for user_idx, group in selected.groupby("user_idx", sort=True):
        user_idx_int = int(user_idx)
        true_items = set(group["item_idx"].astype(int).tolist())

        recommendations = recommend_top_k(
            model,
            user_idx=user_idx_int,
            seen_items=seen_by_user.get(user_idx_int, set()),
            n_items=n_items,
            k=k,
            device=device,
            candidate_batch_size=candidate_batch_size,
        )

        true_item = next(iter(true_items))
        hr_values.append(
            hit_rate_at_k(recommendations, true_item, k)
        )
        ndcg_values.append(
            ndcg_at_k(recommendations, true_item, k)
        )
        recall_values.append(
            recall_at_k(recommendations, true_items, k)
        )

    evaluated = len(hr_values)

    return RankingResult(
        hr=float(np.mean(hr_values)) if evaluated else 0.0,
        ndcg=float(np.mean(ndcg_values)) if evaluated else 0.0,
        recall=float(np.mean(recall_values)) if evaluated else 0.0,
        evaluated_users=evaluated,
    )
