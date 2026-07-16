# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Fraud typology definitions for synthetic graph injection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import append_csv, get_headers, write_output

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

# Description specific to structuring/smurfing patterns.
STRUCTURING_DESCRIPTIONS: list[str] = [
    "cash deposit below reporting threshold",
    "multiple small deposits same day",
    "structured payment just under limit",
    "smurfing deposit via branch teller",
    "incremental cash deposit sub-threshold",
    "repeated near limit ATM deposit",
    "fragmented transfer to evade detection",
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
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        # Allocate every ring's accounts up front from one pool of distinct
        # ids, then give each ring its own slice. Overlapping ranges would
        # merge two rings into a single non-cycle component and make the
        # per-ring involved_accounts labels ambiguous.
        min_d, max_d = self.depth_range
        depths = [random.randint(min_d, max_d) for _ in range(self.num_rings)]
        total_needed = sum(depths)
        if total_needed > max_account_id:
            raise ValueError(
                f"{self.num_rings} fraud rings need {total_needed} distinct "
                f"accounts but only {max_account_id} exist; lower the ring "
                f"count or raise the account scale"
            )
        account_pool = random.sample(range(max_account_id), total_needed)
        pool_offset = 0

        for pattern_id in tqdm(range(self.num_rings), desc="Generating fraud rings"):
            depth = depths[pattern_id]
            ring_ids = account_pool[pool_offset : pool_offset + depth]
            pool_offset += depth

            accounts = [f"acc_{i}" for i in ring_ids]
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

            case_rows.append(
                [
                    f"pat_{pattern_id}",
                    accounts[0],
                    "cycle",
                    depth,
                    involved,
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        write_output(file_tx, headers_tx, tx_rows, compress=compress)
        write_output(file_cases, headers_cases, case_rows, compress=compress)

        return len(tx_rows), current_tx_id


@dataclass
class StructuringGenerator:
    """Generate structuring (smurfing) fraud patterns.
    In a structuring scheme a single coordinator account receives funds from
    several "smurf" accounts, each sending amounts just below the BSA/FinCEN
    Cash Transaction Report (CTR) threshold of $10,000.00. The coordinator
    aggregates these deposits to move a larger sum without triggering a single
    reportable event.

    Graph shape::

        smurf_0 -> coordinator
        smurf_1 -> coordinator
            ...
        smurf_N -> coordinator
    Multiple sources converge on one node.
    This is a structurally distinct from the cyclic ring produced by
    :class:'FraudRingGenerator' and exercises different subgraph-detection
    algorithms.

    Args:
        num_patterns: How many structuring patterns to create.
        smurfs_range: ''(min_smurfs, mac_smurfs)'' - number of feeder
        accounts per pattern. Mirrors the real world practice of using
        3-10 smurfs to stay inconspicuous.
        amount_range: ''(min_amount, max_amount)'' - each smurf transfer is
        drawn uniformly form this range. Defaults to $8_000-$9_900,
        deliberately sub-threshold.
    """

    num_patterns: int = 100
    smurfs_range: tuple[int, int] = (3, 10)
    amount_range: tuple[float, float] = (8_000.00, 9_900.00)
    _descriptions: list[str] = field(default_factory=lambda: STRUCTURING_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> tuple[int, int]:
        """Generate structuring patterns and append to fraud output files.

        Output files are appended to the same ``fraud/`` directory used by
        :class:`FraudRingGenerator` so a single pipeline run can inject both
        typologies into one dataset.

        Args:
            max_account_id: Upper bound of account IDs already generated.
            start_tx_id: First transaction ID to use (must not collide with
                IDs already written by the ring generator or normal txs).
            embedder: Embedding generator instance — same one used by the
                ring generator so embedding provenance is consistent.
            output_dir: Root output directory.
            fmt: ``"csv"`` or ``"neptune"``.
            compress: ZIP the output CSV files.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating structuring patterns"):
            min_s, max_s = self.smurfs_range
            num_smurfs = random.randint(min_s, max_s)

            # The coordinator sits at a random offset; smurfs occupy the
            # num_smurfs slots immediately after it.  We need num_smurfs + 1
            # consecutive IDs so we guard against tiny account pools.
            needed = num_smurfs + 1
            if max_account_id < needed:
                coordinator_idx = 0
            else:
                coordinator_idx = random.randint(0, max_account_id - needed)

            coordinator = f"acc_{coordinator_idx}"
            smurfs = [f"acc_{coordinator_idx + 1 + i}" for i in range(num_smurfs)]
            involved = "|".join([coordinator] + smurfs)

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for smurf in smurfs:
                amount = round(random.uniform(*self.amount_range), 2)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", smurf, coordinator]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, "2024-01-01T12:00:00", desc])
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

            case_rows.append(
                [
                    f"struct_{pattern_id}",
                    coordinator,
                    "structuring",
                    num_smurfs,  # depth = number of feeder hops
                    involved,
                ]
            )

        # Append to the same fraud files so both typologies land in one CSV.
        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id
