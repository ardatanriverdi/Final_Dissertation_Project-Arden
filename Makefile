"""Unit tests for ranking metrics.

Known recommendation lists are used to verify HR@K, NDCG@K and Recall@K.
"""

import pytest

from recommender_benchmark.services.evaluation_service import (
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_hit_rate():
    """Verify hit-rate behaviour for present and absent relevant items."""
    assert hit_rate_at_k([4, 2, 8], 2, 3) == 1.0
    assert hit_rate_at_k([4, 2, 8], 7, 3) == 0.0


def test_ndcg():
    """Verify rank discounting and zero score when no hit occurs."""
    assert ndcg_at_k([2, 4, 8], 2, 3) == 1.0
    assert ndcg_at_k([2, 4, 8], 4, 3) == pytest.approx(
        1.0 / 1.584962500721156
    )
    assert ndcg_at_k([2, 4, 8], 7, 3) == 0.0


def test_recall():
    """Verify recall for partial retrieval and an empty relevant-item set."""
    assert recall_at_k([1, 2, 3], {2, 5}, 3) == 0.5
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0
