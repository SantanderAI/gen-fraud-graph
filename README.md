# gen_fraud_graph

> Synthetic fraud graph generator for training and benchmarking graph-based fraud detection models in financial services.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/gen-fraud-graph.svg)](https://pypi.org/project/gen-fraud-graph/)
[![CI](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/SantanderAI/gen-fraud-graph/branch/main/graph/badge.svg)](https://codecov.io/gh/SantanderAI/gen-fraud-graph)
[![CodeQL](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/SantanderAI/gen-fraud-graph/badge)](https://scorecard.dev/viewer/?uri=github.com/SantanderAI/gen-fraud-graph)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
[![GitHub last commit](https://img.shields.io/github/last-commit/SantanderAI/gen-fraud-graph)](https://github.com/SantanderAI/gen-fraud-graph/commits/main)

---

## Overview

**gen_fraud_graph** is an open-source Python tool that generates massive synthetic financial transaction graphs with injected fraud patterns and optional vector embeddings. It produces CSV datasets ready for ingestion into graph databases (TigerGraph, Neptune, Neo4j, JanusGraph) or for training graph neural networks (GNN).

The generator creates three types of data:
- **Account nodes** — synthetic customer accounts with balance, risk score, and optional embedding vectors
- **Transaction edges** — normal financial transactions between accounts
- **Fraud rings** — cyclic money-laundering patterns with suspicious transaction descriptions

### Key Features

- **Massive scale** — Generate from 1K to 100M+ accounts with configurable scale factor
- **Fraud pattern injection** — Cyclic money-laundering rings with configurable depth (4–7 hops)
- **Parallel generation** — Multi-process workers for fast generation on high-core machines
- **Vector embeddings** — Three providers: `fake` (random, fast), `local` (SentenceTransformers), `openai` (API)
- **Multiple formats** — Generic CSV or AWS Neptune bulk-load format
- **Resume support** — Interrupted generation can resume from where it left off
- **Privacy by design** — All data is 100% synthetic; no real financial data is used

### Use Cases

- Training and evaluating **graph neural networks (GNN)** for fraud detection
- Benchmarking **anti-money laundering (AML)** detection algorithms
- Load-testing graph databases (TigerGraph, Neptune, JanusGraph, NebulaGraph, FalkorDB)
- Research in **financial crime detection** and **anomaly detection** on graphs
- Generating labeled datasets for **deep learning** on graph-structured data

---

## Quick Start

### Installation

```bash
pip install gen-fraud-graph
```

With optional embedding providers:
```bash
pip install 'gen-fraud-graph[local]'    # SentenceTransformers (local model)
pip install 'gen-fraud-graph[openai]'   # OpenAI API embeddings
pip install 'gen-fraud-graph[all]'      # Everything including dev tools
```

Or from source using [uv](https://github.com/astral-sh/uv):
```bash
git clone https://github.com/SantanderAI/gen-fraud-graph.git
cd gen-fraud-graph
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

### CLI Usage

```bash
# Quick test (~1K accounts, ~9K transactions, fake embeddings)
gen-fraud-graph --scale 0.0001 --provider fake --output ./data

# Medium scale (~100K accounts, parallelized)
gen-fraud-graph --scale 0.01 --workers 4 --output ./data

# Full benchmark (~10M accounts, ~90M transactions)
gen-fraud-graph --scale 1.0 --workers 24 --output ./data

# Neptune bulk-load format
gen-fraud-graph --scale 0.01 --format neptune --output ./neptune_data

# Resume interrupted generation (skips completed files)
gen-fraud-graph --scale 1.0 --workers 24 --skip-accounts --output ./data
```

### CLI Arguments

| Flag | Default | Description |
|:---|:---|:---|
| `--scale` | `1.0` | Scale factor. `1.0` = ~10M accounts / ~90M transactions. `0.01` = ~100K accounts. |
| `--provider` | `fake` | Embedding provider: `fake` (random vectors), `local` (SentenceTransformers), `openai`. |
| `--output` | `data` | Output directory for generated CSV files. |
| `--workers` | `1` | Number of parallel worker processes. |
| `--batches` | `1` | Number of file chunks per worker. |
| `--format` | `csv` | Output format: `csv` (generic) or `neptune` (AWS Neptune bulk-load). |
| `--fraud-rings` | auto | Number of fraud rings. Default: auto-scaled from `--scale`. |
| `--compress` | off | ZIP-compress output CSV files. |
| `--skip-accounts` | off | Skip account generation (useful when resuming). |

### Python API

```python
# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from gen_fraud_graph import Config, FraudGraphGenerator

config = Config(
    scale_factor=0.001,         # ~10K accounts, ~90K transactions
    num_fraud_rings=50,         # 50 cyclic fraud patterns
    embedding_provider="fake",  # random vectors (fast, no model needed)
    workers=2,                  # 2 parallel processes
    output_dir="./output",
)

generator = FraudGraphGenerator(config)
generator.run()
```

### Verify Generated Patterns

```bash
python -m gen_fraud_graph.verify --data-dir ./data
```

---

## Output Structure

```
data/
├── accounts/
│   ├── accounts_0_0.csv       # Account nodes (worker 0, batch 0)
│   └── accounts_1_0.csv       # Account nodes (worker 1, batch 0)
├── transactions/
│   ├── transactions_0_0.csv   # Transaction edges (worker 0, batch 0)
│   └── transactions_1_0.csv   # Transaction edges (worker 1, batch 0)
└── fraud/
    ├── transactions_fraud.csv  # Fraud ring transaction edges
    └── fraud_cases.csv         # Fraud ring metadata (pattern_id, accounts, depth)
```

### CSV Schema

**accounts** (`accounts_*.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `account_id` | string | Unique account identifier (`acc_0`, `acc_1`, ...) |
| `customer_name` | string | Synthetic customer name |
| `balance` | float | Account balance (100 – 100,000) |
| `risk_score` | float | Risk score (0.0 – 1.0) |
| `creation_date` | string | Account creation date |

**transactions** (`transactions_*.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `tx_id` | string | Unique transaction identifier |
| `src_id` | string | Source account |
| `dst_id` | string | Destination account |
| `amount` | float | Transaction amount (10 – 500 for normal, 9999 for fraud) |
| `timestamp` | string | Transaction timestamp |
| `description` | string | Transaction description |
| `embedding` | string | Pipe-separated embedding vector |

**fraud_cases** (`fraud/fraud_cases.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `pattern_id` | string | Pattern identifier (`pat_0`, `pat_1`, ...) |
| `start_acc_id` | string | First account in the ring |
| `pattern_type` | string | Always `"cycle"` |
| `depth` | int | Number of hops in the ring (4–7) |
| `involved_accounts` | string | Pipe-separated list of accounts |

---

## Scale Reference

| Scale | Accounts | Transactions | Fraud Rings | Approx. Size |
|:---|:---|:---|:---|:---|
| `0.0001` | 1,000 | 9,000 | 10 | ~2 MB |
| `0.001` | 10,000 | 90,000 | 10 | ~20 MB |
| `0.01` | 100,000 | 900,000 | 10 | ~200 MB |
| `0.1` | 1,000,000 | 9,000,000 | 100 | ~2 GB |
| `1.0` | 10,000,000 | 90,000,000 | 1,000 | ~20 GB |

---

## Project Structure

```
gen_fraud_graph/
├── src/gen_fraud_graph/
│   ├── __init__.py       # Package entry point
│   ├── cli.py            # CLI (gen-fraud-graph command)
│   ├── config.py         # Configuration dataclass
│   ├── embeddings.py     # Embedding providers (fake/local/openai)
│   ├── exporters.py      # CSV/ZIP output writers
│   ├── generator.py      # Core 3-phase pipeline orchestrator
│   ├── typologies.py     # Fraud ring generator
│   └── verify.py         # Pattern verification utility
├── tests/
│   └── test_generator.py # Unit and integration tests
├── examples/
│   └── basic_usage.py    # Minimal Python API example
├── .github/
│   ├── workflows/        # CI (ci, codeql, dep-scan, license-check,
│   │                     #     pattern-check, cla, stale, release)
│   ├── ISSUE_TEMPLATE/   # Bug + feature templates
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── dependabot.yml    # Weekly Python + Actions updates
│   └── pattern-check-allowlist.txt
├── pyproject.toml        # Package metadata and tool config
├── LICENSE               # Apache 2.0
├── NOTICE                # Apache 2.0 attribution
├── CONTRIBUTING.md       # Contribution guidelines
├── CODE_OF_CONDUCT.md    # Contributor Covenant v2.1
├── SECURITY.md           # Vulnerability disclosure policy
├── CODEOWNERS            # Maintainer approvals
└── CHANGELOG.md          # Release history
```

---

## Requirements

Core (always installed):
- Python >= 3.10
- NumPy >= 1.24
- Pandas >= 2.0
- tqdm >= 4.65

Optional:
- `sentence-transformers >= 2.2` — for `--provider local`
- `openai >= 1.0` — for `--provider openai`

---

## Contributing

We welcome contributions from the community. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

By contributing, you agree to the terms of our Contributor License Agreement (CLA).

---

## Security

To report a security vulnerability, please follow the process described in [SECURITY.md](SECURITY.md). **Do not open a public issue for security vulnerabilities.**

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2026 Santander Group
SPDX-License-Identifier: Apache-2.0
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gen_fraud_graph,
  title     = {gen\_fraud\_graph: Synthetic Fraud Graph Generator},
  author    = {Santander AI Lab},
  year      = {2026},
  url       = {https://github.com/SantanderAI/gen-fraud-graph},
  license   = {Apache-2.0}
}
```

---

<!-- GitHub repository metadata (for reference — configured via GitHub UI/API):
  description: "Synthetic fraud graph generator for benchmarking graph-based fraud detection models"
  topics: machine-learning, artificial-intelligence, fraud-detection, graph-neural-network,
          deep-learning, synthetic-data, financial-crime, anti-money-laundering, gnn,
          anomaly-detection, finance, python
  visibility: public
  license: Apache-2.0
  custom_properties:
    category: tool
    track: fast
    status: active
    team: ai-labs
-->
