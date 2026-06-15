from __future__ import annotations

"""End-to-end orchestration for the cross-dataset benchmark.

This module connects the data, training, evaluation and reporting services.
It is the main place to follow the complete experimental workflow.
"""

import time
from pathlib import Path

import pandas as pd
import torch

from recommender_benchmark.config import ExperimentConfig
from recommender_benchmark.models import (
    MatrixFactorizationModel,
    NeuralCollaborativeFilteringModel,
)
from recommender_benchmark.reproducibility import (
    save_run_metadata,
    set_global_seed,
)
from recommender_benchmark.services.data_service import (
    apply_controlled_sparsity,
    dataset_statistics,
    generate_binary_training_data,
    load_raw_dataset,
    prepare_dataset_files,
    prepare_interactions,
)
from recommender_benchmark.services.evaluation_service import (
    evaluate_ranking,
)
from recommender_benchmark.services.reporting_service import (
    create_all_figures,
    save_dataframe,
)
from recommender_benchmark.services.training_service import (
    train_binary_model,
)


def resolve_device() -> torch.device:
    """Use a CUDA GPU when available; otherwise use the CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def prepare_all_datasets(config: ExperimentConfig) -> None:
    """Download and validate every dataset selected in the YAML configuration."""
    config.project.data_dir.mkdir(parents=True, exist_ok=True)

    # Each dataset passes through the same reusable experimental protocol.
    for dataset in config.datasets:
        path = prepare_dataset_files(
            dataset,
            config.project.data_dir,
        )
        print(f"{dataset.name} is ready at {path}")


def run_experiment(
    config: ExperimentConfig,
    *,
    skip_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full reproducible benchmark for all datasets and sparsity levels.
    
    The function prepares data, creates identical training examples, trains both
    models, performs full-catalogue ranking evaluation, saves evidence and plots
    the final dissertation figures.
    """
    set_global_seed(config.project.seed)
    device = resolve_device()

    output_dir = config.project.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_run_metadata(
        output_dir,
        config.raw,
        str(device),
    )

    all_results: list[dict] = []
    all_losses: list[dict] = []
    all_dataset_stats: list[dict] = []
    all_sparsity_stats: list[dict] = []

    total_started = time.perf_counter()

    for dataset in config.datasets:
        if skip_download:
            if dataset.name == "movielens_1m":
                if dataset.extracted_rating_path is None:
                    raise ValueError("MovieLens extracted path missing.")
                data_path = (
                    config.project.data_dir
                    / dataset.extracted_rating_path
                )
            else:
                data_path = (
                    config.project.data_dir
                    / dataset.filename
                )

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Prepared dataset not found: {data_path}"
                )
        else:
            data_path = prepare_dataset_files(
                dataset,
                config.project.data_dir,
            )

        print("\n" + "=" * 80)
        print(f"Preparing {dataset.name}")
        print("=" * 80)

        raw = load_raw_dataset(dataset, data_path)
        prepared = prepare_interactions(
            raw,
            config.preprocessing,
        )

        train = prepared["train"]
        validation = prepared["validation"]
        test = prepared["test"]
        n_users = int(prepared["n_users"])
        n_items = int(prepared["n_items"])

        dataset_dir = output_dir / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        save_dataframe(
            train,
            dataset_dir / "train.csv",
        )
        save_dataframe(
            validation,
            dataset_dir / "validation.csv",
        )
        save_dataframe(
            test,
            dataset_dir / "test.csv",
        )

        for label, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            row = dataset_statistics(
                frame,
                f"{dataset.name}_{label}",
                n_users,
                n_items,
            )
            row["dataset"] = dataset.name
            all_dataset_stats.append(row)

        # Controlled keep rates create the independent sparsity conditions.
        for keep_rate in (
            config.preprocessing.sparsity_keep_rates
        ):
            print("\n" + "-" * 80)
            print(
                f"{dataset.name}: keep_rate={keep_rate:.1f}"
            )
            print("-" * 80)

            sparse_train = apply_controlled_sparsity(
                train,
                keep_rate,
                config.project.seed,
            )

            sparse_stat = dataset_statistics(
                sparse_train,
                f"{dataset.name}_keep_{keep_rate}",
                n_users,
                n_items,
            )
            sparse_stat.update(
                {
                    "dataset": dataset.name,
                    "keep_rate": keep_rate,
                    "sparsity_level": 1.0 - keep_rate,
                }
            )
            all_sparsity_stats.append(sparse_stat)

            binary_train = generate_binary_training_data(
                sparse_train,
                n_items,
                config.preprocessing.negative_ratio,
                config.project.seed,
            )

            # Both models receive the exact same binary training DataFrame.
            model_specs = (
                (
                    "MF",
                    MatrixFactorizationModel(
                        n_users,
                        n_items,
                        config.mf.factors,
                    ),
                    config.mf,
                ),
                (
                    "NCF",
                    NeuralCollaborativeFilteringModel(
                        n_users,
                        n_items,
                        config.ncf.embedding_dim,
                        config.ncf.hidden_layers,
                        config.ncf.dropout,
                    ),
                    config.ncf,
                ),
            )

            # Train and evaluate each model with model-specific hyperparameters.
            for model_name, model, model_config in model_specs:
                set_global_seed(config.project.seed)

                training = train_binary_model(
                    model,
                    binary_train,
                    epochs=model_config.epochs,
                    batch_size=model_config.batch_size,
                    learning_rate=model_config.learning_rate,
                    weight_decay=model_config.weight_decay,
                    device=device,
                    seed=config.project.seed,
                    model_name=model_name,
                )

                for epoch, loss in enumerate(
                    training.losses,
                    start=1,
                ):
                    all_losses.append(
                        {
                            "dataset": dataset.name,
                            "model": model_name,
                            "keep_rate": keep_rate,
                            "sparsity_level": 1.0 - keep_rate,
                            "epoch": epoch,
                            "loss": loss,
                        }
                    )

                ranking = evaluate_ranking(
                    model,
                    train=sparse_train,
                    test=test,
                    n_items=n_items,
                    k=config.evaluation.top_k,
                    device=device,
                    candidate_batch_size=(
                        config.evaluation.candidate_batch_size
                    ),
                    max_eval_users=(
                        config.evaluation.max_eval_users
                    ),
                    seed=config.project.seed,
                )

                all_results.append(
                    {
                        "dataset": dataset.name,
                        "model": model_name,
                        "keep_rate": keep_rate,
                        "sparsity_level": 1.0 - keep_rate,
                        "k": config.evaluation.top_k,
                        "hr": ranking.hr,
                        "ndcg": ranking.ndcg,
                        "recall": ranking.recall,
                        "evaluated_users": (
                            ranking.evaluated_users
                        ),
                        "train_interactions": len(sparse_train),
                        "train_density": sparse_stat["density"],
                        "training_time_seconds": (
                            training.duration_seconds
                        ),
                    }
                )

                print(
                    f"{model_name} HR@{config.evaluation.top_k}: "
                    f"{ranking.hr:.6f}"
                )
                print(
                    f"{model_name} NDCG@{config.evaluation.top_k}: "
                    f"{ranking.ndcg:.6f}"
                )

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    # Convert accumulated evidence into stable tabular output artefacts.
    results = pd.DataFrame(all_results)
    losses = pd.DataFrame(all_losses)
    dataset_stats_frame = pd.DataFrame(all_dataset_stats)
    sparsity_stats_frame = pd.DataFrame(all_sparsity_stats)

    save_dataframe(
        results,
        output_dir / "combined_model_results.csv",
    )
    save_dataframe(
        losses,
        output_dir / "combined_training_losses.csv",
    )
    save_dataframe(
        dataset_stats_frame,
        output_dir / "dataset_statistics.csv",
    )
    save_dataframe(
        sparsity_stats_frame,
        output_dir / "controlled_sparsity_statistics.csv",
    )

    create_all_figures(
        results,
        losses,
        output_dir,
        config.evaluation.top_k,
    )

    print(
        f"\nExperiment completed in "
        f"{time.perf_counter() - total_started:.2f} seconds."
    )

    return results, losses
