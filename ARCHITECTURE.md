[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "recommender-benchmark"
version = "1.0.0"
description = "Reproducible MF vs NCF benchmarking under controlled sparsity"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "numpy==2.1.3",
    "pandas==2.2.3",
    "matplotlib==3.9.2",
    "PyYAML==6.0.2",
    "requests==2.32.3",
    "torch==2.5.1",
]

[project.optional-dependencies]
dev = ["pytest==8.3.4"]

[project.scripts]
recommender-benchmark = "recommender_benchmark.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"
