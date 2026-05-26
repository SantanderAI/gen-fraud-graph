# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for gen_fraud_graph."""

from __future__ import annotations

import argparse
import sys

from gen_fraud_graph import __version__
from gen_fraud_graph.config import Config
from gen_fraud_graph.generator import FraudGraphGenerator


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``gen-fraud-graph`` CLI."""
    parser = argparse.ArgumentParser(
        prog="gen-fraud-graph",
        description="Generate synthetic financial fraud graphs for ML research.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Scale factor (1.0 = ~10M accounts / ~90M transactions, 0.01 = ~100K). Default: 1.0",
    )
    parser.add_argument(
        "--provider", type=str, default="fake",
        choices=["fake", "local", "openai"],
        help="Embedding provider. 'fake' = random vectors (fast, no deps). Default: fake",
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output directory. Default: data",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes. Default: 1",
    )
    parser.add_argument(
        "--batches", type=int, default=1,
        help="Number of file chunks per worker. Default: 1",
    )
    parser.add_argument(
        "--format", type=str, default="csv",
        choices=["csv", "neptune"],
        help="Output format. 'csv' = generic CSV, 'neptune' = AWS Neptune bulk-load. Default: csv",
    )
    parser.add_argument(
        "--fraud-rings", type=int, default=None,
        help="Number of fraud rings to inject. Default: auto (based on scale).",
    )
    parser.add_argument(
        "--compress", action="store_true", default=False,
        help="ZIP-compress output CSV files.",
    )
    parser.add_argument(
        "--skip-accounts", action="store_true", default=False,
        help="Skip account generation (useful when resuming).",
    )

    args = parser.parse_args(argv)

    cfg = Config(
        scale_factor=args.scale,
        num_fraud_rings=args.fraud_rings,
        embedding_provider=args.provider,
        workers=args.workers,
        batches_per_worker=args.batches,
        output_format=args.format,
        compress=args.compress,
        output_dir=args.output,
    )

    generator = FraudGraphGenerator(cfg)
    generator.run(skip_accounts=args.skip_accounts)


if __name__ == "__main__":
    main()
