"""Unit tests for preprocessing, chronological splitting and sparsity logic.

Small deterministic fixtures verify core behaviour without downloading the
research datasets, allowing the tests to run quickly in GitHub Actions.
"""

import pandas as pd

from recommender_benchmark.services.data_service import (
    apply_controlled_sparsity,
    chronological_leave_one_out,
    convert_to_implicit,
    encode_identifiers,
)


def sample_raw() -> pd.DataFrame:
    """Create a small deterministic ratings table for unit tests."""
    rows = []

    for user in ("u1", "u2"):
        for index, item in enumerate(("i1", "i2", "i3", "i4")):
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "rating": 5 if index != 1 else 3,
                    "timestamp": index + 1,
                }
            )

    return pd.DataFrame(rows)


def test_convert_to_implicit_filters_ratings():
    """Verify that the rating threshold removes lower ratings."""
    implicit = convert_to_implicit(sample_raw(), 4.0)

    assert set(implicit["interaction"]) == {1}
    assert len(implicit) == 6


def test_leave_one_out_produces_one_validation_and_test_per_user():
    """Verify chronological leave-one-out sizes."""
    implicit = convert_to_implicit(sample_raw(), 3.0)
    encoded, _, _ = encode_identifiers(implicit)
    train, validation, test = chronological_leave_one_out(encoded)

    assert len(validation) == 2
    assert len(test) == 2
    assert train["user_idx"].nunique() == 2


def test_controlled_sparsity_keeps_each_user():
    """Verify that controlled sparsity retains every user."""
    implicit = convert_to_implicit(sample_raw(), 3.0)
    encoded, _, _ = encode_identifiers(implicit)
    train, _, _ = chronological_leave_one_out(encoded)

    sparse = apply_controlled_sparsity(
        train,
        keep_rate=0.1,
        seed=42,
    )

    assert sparse["user_idx"].nunique() == train["user_idx"].nunique()
    assert sparse.groupby("user_idx").size().min() >= 1
