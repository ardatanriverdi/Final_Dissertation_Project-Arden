# Recommender Benchmark: MF vs NCF Under Controlled Sparsity

A reproducible, cross-dataset benchmark comparing:

- Matrix Factorization (MF)
- Neural Collaborative Filtering (NCF)

Datasets:

- MovieLens 1M
- Amazon Digital Music 5-core

The project uses the same preprocessing, negative sampling, controlled sparsity, ranking metrics and random seed for both datasets.

## Architecture

The repository uses a service-oriented modular structure:

- **Data service:** download, validate, load, preprocess, split, sparsify and sample negatives.
- **Training service:** deterministic PyTorch training for MF and NCF.
- **Evaluation service:** HR@10, NDCG@10 and Recall@10.
- **Reporting service:** CSV outputs, run metadata and dissertation figures.
- **Pipeline:** orchestrates the complete experiment.

This is intentionally modular rather than a network-distributed microservice system. It provides microservice-style separation of responsibilities without adding unnecessary deployment complexity to a dissertation artefact.

## Project structure

```text
.
├── configs/default.yaml
├── src/recommender_benchmark/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── reproducibility.py
│   ├── models/
│   └── services/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Local installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
pip install -e .
```

## Commands

Download and prepare datasets:

```bash
recommender-benchmark prepare --config configs/default.yaml
```

Run the full experiment:

```bash
recommender-benchmark run --config configs/default.yaml
```

Run a quick smoke test:

```bash
recommender-benchmark run --config configs/default.yaml --smoke-test
```

Run tests:

```bash
pytest
```

## Docker

Build:

```bash
docker build -t recommender-benchmark .
```

Run the full benchmark:

```bash
docker run --rm \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/outputs:/app/outputs" \
  recommender-benchmark
```

Windows PowerShell:

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/outputs:/app/outputs" `
  recommender-benchmark
```

## Docker Compose

Run the data preparation service followed by the experiment service:

```bash
docker compose up --build --abort-on-container-exit
```

Run only the tests service:

```bash
docker compose run --rm tests
```

## GitHub workflow

The included GitHub Actions workflow:

- installs the package;
- checks that the source compiles;
- runs unit tests;
- avoids downloading the full research datasets in CI.

## Reproducibility outputs

Each run saves:

- a copy of the effective YAML configuration;
- environment and package versions;
- Git commit hash when available;
- dataset statistics;
- controlled sparsity statistics;
- training loss values;
- model ranking results;
- training times;
- dissertation-ready PNG figures.

## Default experimental protocol

- Random seed: 42
- Implicit threshold: rating >= 4
- Negative samples per positive interaction: 4
- Sparsity keep rates: 1.0, 0.7, 0.5, 0.3, 0.1
- Ranking cut-off: K = 10
- MF epochs: 10
- NCF epochs: 5
- Leave-one-out chronological split: one validation item and one test item per user

Because there is one relevant test item per user, HR@10 and Recall@10 are numerically identical in this protocol. Both are retained for consistency with the dissertation design.
