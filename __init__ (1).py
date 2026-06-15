from __future__ import annotations

"""Command-line interface for preparing datasets and running experiments.

The CLI provides one stable entry point for local execution, Docker,
Docker Compose and automated workflows.
"""

import argparse
import sys
from pathlib import Path

from recommender_benchmark.config import (
    apply_smoke_test,
    load_config,
)
from recommender_benchmark.pipeline import (
    prepare_all_datasets,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    """Define the prepare and run command-line interfaces."""
    parser = argparse.ArgumentParser(
        prog="recommender-benchmark",
        description=(
            "Run the MF vs NCF controlled-sparsity benchmark."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    prepare = subparsers.add_parser(
        "prepare",
        help="Download and validate all configured datasets.",
    )
    prepare.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )

    run = subparsers.add_parser(
        "run",
        help="Run the complete experiment.",
    )
    run.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    run.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already prepared dataset files.",
    )
    run.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one sparsity level, one epoch and 100 users."
        ),
    )

    return parser


def main() -> int:
    """Parse command-line arguments and call the requested pipeline operation."""
    parser = build_parser()
    arguments = parser.parse_args()
    config = load_config(arguments.config)

    if arguments.command == "prepare":
        prepare_all_datasets(config)
        return 0

    if arguments.smoke_test:
        config = apply_smoke_test(config)

    run_experiment(
        config,
        skip_download=arguments.skip_download,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
