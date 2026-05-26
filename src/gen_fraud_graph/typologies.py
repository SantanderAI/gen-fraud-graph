# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Fraud typology definitions for synthetic graph injection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import get_headers, write_output


# ---------------------------------------------------------------------------
# Suspicious transaction descriptions used across typologies
# ---------------------------------------------------------------------------

SUSPICIOUS_DESCRIPTIONS: list[str] = [
    "offshore transfer to tax haven",
    "structuring deposit below threshold",
    "rapid movement of funds between accounts",
    "shell company payment",
    "layered transfer via intermediary",
    "round-trip transaction",
    "dormant account sudden activity",
    "high-value cross-border wire",
]


# ---------------------------------------------------------------------------
# Fraud ring generator (cyclic money-laundering patterns)
# ---------------------------------------------------------------------------

@dataclass
class FraudRingGenerator:
    """Generate cyclic fraud-ring patterns.

    Each ring is a cycle of ``depth`` accounts connected by suspicious
    high-value transactions.

    Args:
        num_rings: How many rings to create.
        depth_range: ``(min_depth, max_depth)`` hops per ring.
        amount: Fixed transaction amount injected in fraud edges.
    """

    num_rings: int = 100
    depth_range: tuple[int, int] = (4, 7)
    amount: float = 9999.00
    _descriptions: list[str] = field(default_factory=lambda: SUSPICIOUS_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> tuple[int, int]:
        """Generate fraud rings and write output files.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os
        from tqdm import tqdm

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id", "start_acc_id", "pattern_type", "depth", "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_rings), desc="Generating fraud rings"):
            min_d, max_d = self.depth_range
            depth = random.randint(min_d, max_d)

            # Pick a contiguous range of accounts for the ring
            if max_account_id < depth + 1:
                start_node = 0
            else:
                start_node = random.randint(0, max_account_id - depth - 1)

            accounts = [f"acc_{start_node + d}" for d in range(depth)]
            involved = "|".join(accounts)

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for k in range(depth):
                src = accounts[k]
                dst = accounts[(k + 1) % depth]
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", src, dst]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([self.amount, "2024-01-01T12:00:00", desc])
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append([
                f"pat_{pattern_id}", accounts[0], "cycle", depth, involved,
            ])

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        write_output(file_tx, headers_tx, tx_rows, compress=compress)
        write_output(file_cases, headers_cases, case_rows, compress=compress)

        return len(tx_rows), current_tx_id
