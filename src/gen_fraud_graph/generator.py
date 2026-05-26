# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Core generator — orchestrates account, transaction, and fraud generation."""

from __future__ import annotations

import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

import numpy as np
from tqdm import tqdm

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import get_headers
from gen_fraud_graph.typologies import FraudRingGenerator


# ---------------------------------------------------------------------------
# Normal transaction descriptions
# ---------------------------------------------------------------------------

NORMAL_DESCRIPTIONS: list[str] = [
    "grocery store purchase",
    "salary deposit",
    "utility bill payment",
    "online subscription",
    "restaurant payment",
    "atm withdrawal",
    "peer to peer transfer",
    "insurance premium",
    "mortgage payment",
    "investment deposit",
]


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------

def _generate_accounts_chunk(
    worker_id: int,
    batch_id: int,
    start_id: int,
    count: int,
    provider: str,
    dim: int,
    output_dir: str,
    fmt: str = "csv",
) -> str:
    """Generate a chunk of account rows (called by ProcessPoolExecutor)."""
    random.seed(start_id)
    embedder = EmbeddingGenerator(provider, dim=dim)

    acc_dir = os.path.join(output_dir, "accounts")
    os.makedirs(acc_dir, exist_ok=True)

    headers = get_headers("account", fmt)  # type: ignore[arg-type]
    csv_path = os.path.join(acc_dir, f"accounts_{worker_id}_{batch_id}.csv")

    # Resume support
    existing_rows = 0
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, "r") as fh:
            existing_rows = sum(1 for _ in fh) - 1
        if existing_rows >= count:
            return f"Worker {worker_id} Batch {batch_id}: Skipped (already complete)"
        print(f"  Worker {worker_id} Batch {batch_id}: Resuming from row {existing_rows}")

    batch_size = 5_000
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(headers)

        for i in range(max(0, existing_rows), count, batch_size):
            chunk_count = min(batch_size, count - i)
            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for j in range(chunk_count):
                uid = start_id + i + j
                aid = f"acc_{uid}"
                name = f"Customer_{uid}"
                batch_texts.append(name)

                row: list = [
                    aid,
                    name,
                    round(random.uniform(100, 100_000), 2),
                    round(random.uniform(0, 1), 4),
                    "2023-01-01",
                ]
                if fmt == "neptune":
                    row.insert(1, "Account")
                batch_rows.append(row)

            if fmt == "neptune":
                embeddings = embedder.generate(batch_texts)
                final_rows = []
                for idx, r in enumerate(batch_rows):
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    final_rows.append(r + [";".join(map(str, vec))])
            else:
                final_rows = batch_rows

            writer.writerows(final_rows)
            if (i + chunk_count) % 50_000 == 0:
                print(f"  Worker {worker_id} Batch {batch_id}: {i + chunk_count} accounts written")

    return f"Worker {worker_id} Batch {batch_id}: Generated {count} accounts"


def _generate_transactions_chunk(
    worker_id: int,
    batch_id: int,
    start_tx_id: int,
    count: int,
    total_accounts: int,
    provider: str,
    dim: int,
    output_dir: str,
    fmt: str = "csv",
) -> str:
    """Generate a chunk of transaction rows (called by ProcessPoolExecutor)."""
    random.seed(start_tx_id)
    embedder = EmbeddingGenerator(provider, dim=dim)

    tx_dir = os.path.join(output_dir, "transactions")
    os.makedirs(tx_dir, exist_ok=True)

    headers = get_headers("transaction", fmt)  # type: ignore[arg-type]
    csv_path = os.path.join(tx_dir, f"transactions_{worker_id}_{batch_id}.csv")

    # Resume support
    existing_rows = 0
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, "r") as fh:
            existing_rows = sum(1 for _ in fh) - 1
        if existing_rows >= count:
            return f"Worker {worker_id} Batch {batch_id}: Skipped (already complete)"
        print(f"  Worker {worker_id} Batch {batch_id}: Resuming from row {existing_rows}")

    embed_batch_size = 5_000
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(headers)

        for i in range(max(0, existing_rows), count, embed_batch_size):
            chunk_count = min(embed_batch_size, count - i)
            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for j in range(chunk_count):
                tx_uid = start_tx_id + i + j
                src = f"acc_{random.randint(0, total_accounts - 1)}"
                dst = f"acc_{random.randint(0, total_accounts - 1)}"
                while src == dst:
                    dst = f"acc_{random.randint(0, total_accounts - 1)}"

                desc = random.choice(NORMAL_DESCRIPTIONS)
                batch_texts.append(desc)

                row: list = [
                    f"tx_{tx_uid}",
                    src,
                    dst,
                    round(random.uniform(10, 500), 2),
                    "2024-01-01T10:00:00",
                    desc,
                ]
                if fmt == "neptune":
                    row.insert(3, "TRANSFER")
                batch_rows.append(row)

            embeddings = embedder.generate(batch_texts)

            final_rows: list[list] = []
            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    final_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    final_rows.append(r + ["|".join(map(str, vec))])

            writer.writerows(final_rows)
            if (i + chunk_count) % 50_000 == 0:
                print(
                    f"  Worker {worker_id} Batch {batch_id}: "
                    f"{i + chunk_count} transactions written"
                )

    return f"Worker {worker_id} Batch {batch_id}: Generated {count} transactions"


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

class FraudGraphGenerator:
    """Orchestrates the full synthetic fraud-graph generation pipeline.

    Usage::

        from gen_fraud_graph import FraudGraphGenerator, Config

        cfg = Config(scale_factor=0.01, embedding_provider="fake")
        gen = FraudGraphGenerator(cfg)
        gen.run()

    The output directory will contain:

    * ``accounts/``  — account node CSVs (one per worker × batch)
    * ``transactions/`` — legitimate transaction edge CSVs
    * ``fraud/`` — ``transactions_fraud.csv`` and ``fraud_cases.csv``
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def run(self, *, skip_accounts: bool = False) -> None:
        """Execute the three-phase generation pipeline.

        Args:
            skip_accounts: If *True*, skip Phase 1 (useful when resuming).
        """
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        print("=" * 50)
        print("gen_fraud_graph — Synthetic Fraud Graph Generator")
        print("=" * 50)
        print(f"  Scale factor : {cfg.scale_factor}")
        print(f"  Accounts     : {cfg.num_accounts:,}")
        print(f"  Transactions : {cfg.num_transactions:,}")
        print(f"  Fraud rings  : {cfg.num_fraud_rings:,}")
        print(f"  Format       : {cfg.output_format}")
        print(f"  Embedding    : {cfg.embedding_provider}")
        print(f"  Workers      : {cfg.workers}")
        print(f"  Compress     : {cfg.compress}")
        print(f"  Output       : {cfg.output_dir}")
        print("=" * 50)

        # Phase 1 — Accounts
        if not skip_accounts:
            self._generate_accounts()
        else:
            print("\n[Phase 1] Skipping accounts (--skip-accounts)")

        # Phase 2 — Transactions
        self._generate_transactions()

        # Phase 3 — Fraud rings
        self._generate_fraud()

        print("\nDone! All data generated.")

    # ------------------------------------------------------------------

    def _generate_accounts(self) -> None:
        cfg = self.cfg
        print("\n[Phase 1] Generating accounts...")

        acc_per_worker = cfg.num_accounts // cfg.workers
        acc_per_batch = acc_per_worker // cfg.batches_per_worker

        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = []
            for w in range(cfg.workers):
                for b in range(cfg.batches_per_worker):
                    global_idx = w * cfg.batches_per_worker + b
                    start_id = global_idx * acc_per_batch
                    futures.append(
                        pool.submit(
                            _generate_accounts_chunk,
                            w, b, start_id, acc_per_batch,
                            cfg.embedding_provider, cfg.embedding_dim,
                            cfg.output_dir, cfg.output_format,
                        )
                    )
            for f in tqdm(futures, total=len(futures), desc="Account batches"):
                f.result()

    def _generate_transactions(self) -> None:
        cfg = self.cfg
        print("\n[Phase 2] Generating transactions...")

        tx_per_worker = cfg.num_transactions // cfg.workers
        tx_per_batch = tx_per_worker // cfg.batches_per_worker

        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = []
            for w in range(cfg.workers):
                for b in range(cfg.batches_per_worker):
                    global_idx = w * cfg.batches_per_worker + b
                    start_id = global_idx * tx_per_batch
                    futures.append(
                        pool.submit(
                            _generate_transactions_chunk,
                            w, b, start_id, tx_per_batch,
                            cfg.num_accounts, cfg.embedding_provider,
                            cfg.embedding_dim, cfg.output_dir,
                            cfg.output_format,
                        )
                    )
            for f in tqdm(futures, total=len(futures), desc="Transaction batches"):
                f.result()

    def _generate_fraud(self) -> None:
        cfg = self.cfg
        print("\n[Phase 3] Generating fraud rings...")

        embedder = EmbeddingGenerator(cfg.embedding_provider, dim=cfg.embedding_dim)
        fraud_gen = FraudRingGenerator(
            num_rings=cfg.num_fraud_rings,
            depth_range=cfg.fraud_ring_depth_range,
        )
        n_tx, _ = fraud_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=cfg.num_transactions,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(f"  Injected {n_tx:,} fraud transactions across {cfg.num_fraud_rings:,} rings")
