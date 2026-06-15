from __future__ import annotations

"""Data acquisition, validation, preprocessing and sampling service.

Responsibilities include downloading both datasets, checking archives,
converting ratings to implicit feedback, filtering, encoding identifiers,
chronological splitting, controlled sparsity and negative sampling.
"""

import gzip
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from recommender_benchmark.config import (
    DatasetConfig,
    PreprocessingConfig,
)


def build_http_session() -> requests.Session:
    """
    Create an HTTP session with retry and exponential backoff behaviour.
    
    Retries reduce failures caused by temporary server errors or rate limiting.
    """
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))

    return session


def download_file(
    urls: tuple[str, ...],
    destination: Path,
    timeout: int = 45,
) -> None:
    """
    Download from the first working URL into a temporary .part file.
    
    A partial file is renamed only after a successful response, preventing an
    interrupted download from being mistaken for a valid dataset archive.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    errors: list[str] = []

    session = build_http_session()

    # Try each configured mirror in order until one download completes.
    for url in urls:
        try:
            print(f"Downloading {url}")
            with session.get(
                url,
                stream=True,
                timeout=(15, timeout),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()

                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            # Atomic replacement exposes the final filename only after success.
            temporary.replace(destination)
            print(
                f"Saved {destination} "
                f"({destination.stat().st_size:,} bytes)"
            )
            return

        except (requests.RequestException, OSError) as exc:
            errors.append(f"{url}: {exc}")
            temporary.unlink(missing_ok=True)

    raise RuntimeError(
        "All dataset download URLs failed:\n" + "\n".join(errors)
    )


def validate_movie_lens_zip(path: Path) -> None:
    """Verify that the MovieLens download is a readable ZIP archive."""
    if not zipfile.is_zipfile(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{path} is not a valid ZIP archive. "
            "The incomplete file was removed."
        )


def validate_gzip(path: Path) -> None:
    """Verify that the Amazon download is a readable gzip archive."""
    try:
        with gzip.open(path, "rb") as handle:
            handle.read(1)
    except (OSError, EOFError) as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{path} is not a valid gzip archive. "
            "The incomplete file was removed."
        ) from exc


def prepare_dataset_files(
    dataset: DatasetConfig,
    data_dir: Path,
) -> Path:
    """
    Download, validate and, where necessary, extract one configured dataset.
    
    The function returns the exact local file that the dataset loader should read.
    """
    archive_path = data_dir / dataset.filename

    if dataset.name == "movielens_1m":
        if dataset.extracted_rating_path is None:
            raise ValueError("MovieLens extracted path is missing.")

        rating_path = data_dir / dataset.extracted_rating_path

        if rating_path.exists():
            return rating_path

        if not archive_path.exists():
            download_file(dataset.urls, archive_path)

        validate_movie_lens_zip(archive_path)

        extract_dir = data_dir / "ml-1m"
        if extract_dir.exists() and not rating_path.exists():
            shutil.rmtree(extract_dir)

        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(data_dir)

        if not rating_path.exists():
            raise RuntimeError(
                f"Expected MovieLens file was not extracted: {rating_path}"
            )

        return rating_path

    if dataset.name == "amazon_digital_music":
        if not archive_path.exists():
            download_file(dataset.urls, archive_path)

        validate_gzip(archive_path)
        return archive_path

    raise ValueError(f"Unsupported dataset: {dataset.name}")


def load_raw_dataset(
    dataset: DatasetConfig,
    data_path: Path,
) -> pd.DataFrame:
    """
    Load MovieLens or Amazon data into a common four-column rating schema.
    
    Normalising both sources here allows all later preprocessing code to be reused.
    """
    if dataset.name == "movielens_1m":
        return pd.read_csv(
            data_path,
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )

    if dataset.name == "amazon_digital_music":
        records: list[dict[str, Any]] = []

        with gzip.open(data_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                records.append(
                    {
                        "user_id": row.get("reviewerID"),
                        "item_id": row.get("asin"),
                        "rating": row.get("overall"),
                        "timestamp": row.get("unixReviewTime"),
                    }
                )

        return pd.DataFrame.from_records(records).dropna()

    raise ValueError(f"Unsupported dataset: {dataset.name}")


def convert_to_implicit(
    raw: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Keep ratings at or above the threshold and mark them as positive interactions.
    
    Ratings below the threshold are not treated as explicit negatives; negatives
    are sampled later from unobserved user-item pairs.
    """
    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required.difference(raw.columns)

    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")

    interactions = raw.loc[
        raw["rating"] >= threshold,
        ["user_id", "item_id", "timestamp"],
    ].copy()

    interactions["interaction"] = 1

    return interactions[
        ["user_id", "item_id", "interaction", "timestamp"]
    ].reset_index(drop=True)


def filter_min_interactions(
    interactions: pd.DataFrame,
    minimum_users: int,
    minimum_items: int,
) -> pd.DataFrame:
    """
    Iteratively remove users and items below the minimum interaction counts.
    
    Iteration is required because removing weak users can make some items fall
    below the threshold, and removing those items can then affect users again.
    """
    filtered = interactions.copy()

    # Repeat the user/item filtering cycle until no additional rows are removed.
    while True:
        previous_size = len(filtered)

        user_counts = filtered.groupby("user_id")["item_id"].size()
        valid_users = user_counts[
            user_counts >= minimum_users
        ].index
        filtered = filtered[
            filtered["user_id"].isin(valid_users)
        ]

        item_counts = filtered.groupby("item_id")["user_id"].size()
        valid_items = item_counts[
            item_counts >= minimum_items
        ].index
        filtered = filtered[
            filtered["item_id"].isin(valid_items)
        ]

        if len(filtered) == previous_size:
            return filtered.reset_index(drop=True)


def encode_identifiers(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[Any, int], dict[Any, int]]:
    """
    Map original user and item identifiers to contiguous zero-based indices.
    
    Contiguous indices are required by PyTorch embedding layers.
    """
    encoded = interactions.copy()

    user_values = sorted(encoded["user_id"].unique())
    item_values = sorted(encoded["item_id"].unique())

    user_map = {
        value: index
        for index, value in enumerate(user_values)
    }
    item_map = {
        value: index
        for index, value in enumerate(item_values)
    }

    encoded["user_idx"] = encoded["user_id"].map(user_map)
    encoded["item_idx"] = encoded["item_id"].map(item_map)

    return encoded, user_map, item_map


def chronological_leave_one_out(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign each user's latest interaction to test and second latest to validation.
    
    All earlier interactions form the training set, preventing future interactions
    from leaking into the training history.
    """
    ordered = interactions.sort_values(
        ["user_idx", "timestamp", "item_idx"]
    ).copy()

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    # Split independently per user so every user contributes chronologically.
    for _, group in ordered.groupby("user_idx", sort=True):
        if len(group) < 3:
            continue

        train_parts.append(group.iloc[:-2])
        validation_parts.append(group.iloc[-2:-1])
        test_parts.append(group.iloc[-1:])

    if not train_parts:
        raise ValueError("No users have enough interactions for splitting.")

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(validation_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def dataset_statistics(
    frame: pd.DataFrame,
    label: str,
    total_users: int,
    total_items: int,
) -> dict[str, float | int | str]:
    """Calculate interaction count, density and sparsity on a fixed user-item space."""
    interactions = len(frame)
    density = (
        interactions / (total_users * total_items)
        if total_users and total_items
        else 0.0
    )

    return {
        "label": label,
        "users": total_users,
        "items": total_items,
        "interactions": interactions,
        "density": density,
        "sparsity": 1.0 - density,
    }


def apply_controlled_sparsity(
    train: pd.DataFrame,
    keep_rate: float,
    seed: int,
) -> pd.DataFrame:
    """
    Retain a controlled proportion of each user's training history.
    
    At least one training interaction is preserved for every user so all test users
    remain representable by the recommender models.
    """
    if not 0 < keep_rate <= 1:
        raise ValueError("keep_rate must be in the interval (0, 1].")

    if keep_rate == 1:
        return train.copy().reset_index(drop=True)

    # Derive a deterministic but condition-specific random stream.
    rng = np.random.default_rng(
        seed + int(round(keep_rate * 10_000))
    )
    sampled_parts: list[pd.DataFrame] = []

    for _, group in train.groupby("user_idx", sort=True):
        keep_count = max(1, int(np.ceil(len(group) * keep_rate)))
        selected_indices = rng.choice(
            group.index.to_numpy(),
            size=keep_count,
            replace=False,
        )
        sampled_parts.append(train.loc[selected_indices])

    return (
        pd.concat(sampled_parts)
        .sort_values(["user_idx", "timestamp", "item_idx"])
        .reset_index(drop=True)
    )


def build_seen_items(
    interactions: pd.DataFrame,
) -> dict[int, set[int]]:
    """Build a user-to-items lookup used by sampling and recommendation filtering."""
    return {
        int(user_idx): set(group["item_idx"].astype(int).tolist())
        for user_idx, group in interactions.groupby("user_idx")
    }


def generate_binary_training_data(
    train: pd.DataFrame,
    n_items: int,
    negative_ratio: int,
    seed: int,
) -> pd.DataFrame:
    """
    Combine observed positive pairs with sampled unobserved negative pairs.
    
    The same generated DataFrame is supplied to MF and NCF within a condition,
    ensuring that differences are caused by the models rather than training samples.
    """
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")

    rng = np.random.default_rng(seed)
    positive = train[["user_idx", "item_idx"]].copy()
    positive["label"] = 1.0

    seen = build_seen_items(train)
    all_items = np.arange(n_items, dtype=np.int64)
    negative_frames: list[pd.DataFrame] = []

    for user_idx, group in train.groupby("user_idx", sort=True):
        user_idx_int = int(user_idx)
        available = np.setdiff1d(
            all_items,
            np.fromiter(seen[user_idx_int], dtype=np.int64),
            assume_unique=False,
        )

        requested = len(group) * negative_ratio
        if len(available) == 0:
            continue

        # Avoid duplicate negatives unless the unseen catalogue is too small.
        replace = requested > len(available)
        sampled = rng.choice(
            available,
            size=requested,
            replace=replace,
        )

        negative_frames.append(
            pd.DataFrame(
                {
                    "user_idx": user_idx_int,
                    "item_idx": sampled,
                    "label": 0.0,
                }
            )
        )

    negatives = (
        pd.concat(negative_frames, ignore_index=True)
        if negative_frames
        else pd.DataFrame(
            columns=["user_idx", "item_idx", "label"]
        )
    )

    binary = pd.concat(
        [positive, negatives],
        ignore_index=True,
    )

    return binary.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def prepare_interactions(
    raw: pd.DataFrame,
    preprocessing: PreprocessingConfig,
):
    """
    Run the complete shared preprocessing sequence and return all prepared objects.
    
    The returned dictionary contains encoded interactions, data splits, mappings
    and user/item cardinalities.
    """
    implicit = convert_to_implicit(
        raw,
        preprocessing.implicit_threshold,
    )
    filtered = filter_min_interactions(
        implicit,
        preprocessing.min_user_interactions,
        preprocessing.min_item_interactions,
    )
    encoded, user_map, item_map = encode_identifiers(filtered)
    train, validation, test = chronological_leave_one_out(encoded)

    return {
        "interactions": encoded,
        "train": train,
        "validation": validation,
        "test": test,
        "user_map": user_map,
        "item_map": item_map,
        "n_users": len(user_map),
        "n_items": len(item_map),
    }
