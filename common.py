from __future__ import annotations

"""Configuration models and YAML loading utilities.

All experimental settings are defined outside the training code so that
a complete run can be reproduced from a saved configuration file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Store project-wide paths, name and random seed."""
    name: str
    data_dir: Path
    output_dir: Path
    seed: int


@dataclass(frozen=True)
class DatasetConfig:
    """Describe one downloadable dataset and its possible source URLs."""
    name: str
    filename: str
    urls: tuple[str, ...]
    extracted_rating_path: str | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    """Store implicit-feedback, filtering, sparsity and sampling settings."""
    implicit_threshold: float
    min_user_interactions: int
    min_item_interactions: int
    sparsity_keep_rates: tuple[float, ...]
    negative_ratio: int


@dataclass(frozen=True)
class EvaluationConfig:
    """Store top-K and candidate-scoring evaluation settings."""
    top_k: int
    max_eval_users: int | None
    candidate_batch_size: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Store hyperparameters shared by trainable PyTorch models."""
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int


@dataclass(frozen=True)
class MFConfig(ModelTrainingConfig):
    """Extend shared training settings with the number of latent factors."""
    factors: int


@dataclass(frozen=True)
class NCFConfig(ModelTrainingConfig):
    """Extend shared training settings with neural architecture parameters."""
    embedding_dim: int
    hidden_layers: tuple[int, ...]
    dropout: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Aggregate every setting required for one reproducible experiment."""
    project: ProjectConfig
    datasets: tuple[DatasetConfig, ...]
    preprocessing: PreprocessingConfig
    evaluation: EvaluationConfig
    mf: MFConfig
    ncf: NCFConfig
    raw: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    """
    Load a YAML file and convert its nested values into typed configuration objects.
    
    Typed configuration objects make invalid or missing settings easier to identify
    before a long model-training run begins.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    project_raw = raw["project"]
    project = ProjectConfig(
        name=str(project_raw["name"]),
        data_dir=Path(project_raw["data_dir"]),
        output_dir=Path(project_raw["output_dir"]),
        seed=int(project_raw["seed"]),
    )

    selected_names = tuple(raw["datasets"]["selected"])
    dataset_configs: list[DatasetConfig] = []

    for name in selected_names:
        item = raw["datasets"][name]
        dataset_configs.append(
            DatasetConfig(
                name=name,
                filename=str(item["filename"]),
                urls=tuple(str(url) for url in item["urls"]),
                extracted_rating_path=item.get("extracted_rating_path"),
            )
        )

    prep_raw = raw["preprocessing"]
    preprocessing = PreprocessingConfig(
        implicit_threshold=float(prep_raw["implicit_threshold"]),
        min_user_interactions=int(prep_raw["min_user_interactions"]),
        min_item_interactions=int(prep_raw["min_item_interactions"]),
        sparsity_keep_rates=tuple(float(x) for x in prep_raw["sparsity_keep_rates"]),
        negative_ratio=int(prep_raw["negative_ratio"]),
    )

    eval_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        top_k=int(eval_raw["top_k"]),
        max_eval_users=(
            None
            if eval_raw.get("max_eval_users") is None
            else int(eval_raw["max_eval_users"])
        ),
        candidate_batch_size=int(eval_raw["candidate_batch_size"]),
    )

    mf_raw = raw["models"]["matrix_factorization"]
    mf = MFConfig(
        factors=int(mf_raw["factors"]),
        learning_rate=float(mf_raw["learning_rate"]),
        weight_decay=float(mf_raw["weight_decay"]),
        batch_size=int(mf_raw["batch_size"]),
        epochs=int(mf_raw["epochs"]),
    )

    ncf_raw = raw["models"]["neural_collaborative_filtering"]
    ncf = NCFConfig(
        embedding_dim=int(ncf_raw["embedding_dim"]),
        hidden_layers=tuple(int(x) for x in ncf_raw["hidden_layers"]),
        dropout=float(ncf_raw["dropout"]),
        learning_rate=float(ncf_raw["learning_rate"]),
        weight_decay=float(ncf_raw["weight_decay"]),
        batch_size=int(ncf_raw["batch_size"]),
        epochs=int(ncf_raw["epochs"]),
    )

    return ExperimentConfig(
        project=project,
        datasets=tuple(dataset_configs),
        preprocessing=preprocessing,
        evaluation=evaluation,
        mf=mf,
        ncf=ncf,
        raw=raw,
    )


def apply_smoke_test(config: ExperimentConfig) -> ExperimentConfig:
    """
    Create a temporary lightweight configuration for fast pipeline verification.
    
    The smoke test uses one sparsity level, one epoch and a small evaluation sample.
    It checks integration only and must not be used as a final dissertation result.
    """
    raw = dict(config.raw)
    raw["preprocessing"] = dict(raw["preprocessing"])
    raw["evaluation"] = dict(raw["evaluation"])
    raw["models"] = {
        key: dict(value)
        for key, value in raw["models"].items()
    }

    raw["preprocessing"]["sparsity_keep_rates"] = [1.0]
    raw["evaluation"]["max_eval_users"] = 100
    raw["models"]["matrix_factorization"]["epochs"] = 1
    raw["models"]["neural_collaborative_filtering"]["epochs"] = 1

    temp_path = config.project.output_dir / "_smoke_config.yaml"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)

    return load_config(temp_path)
