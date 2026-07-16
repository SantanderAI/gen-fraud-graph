# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Fraud typology definitions for synthetic graph injection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

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

# Descriptions used by decoy LEGITIMATE high-value cycles.  These are normal,
# benign business flows that happen to form a cycle and carry a large amount.
# Their purpose is to defeat pure amount-thresholding (they look high-value)
# and pure cycle-topology (they are cycles) without being fraud.
DECOY_DESCRIPTIONS: list[str] = [
    "supplier invoice settlement",
    "intercompany treasury sweep",
    "payroll batch funding",
    "merchant settlement payout",
    "loan disbursement",
    "dividend distribution",
    "property purchase completion",
    "fleet lease payment",
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
        amount_jitter: Relative +/- jitter applied to *amount* per fraud edge.
            ``0.0`` (the default) keeps the exact sentinel on every edge, so a
            naive amount threshold catches all fraud.  A value like ``0.5``
            spreads fraud amounts over ``amount * [0.5, 1.5]`` so the sentinel
            is no longer a giveaway.
        ring_overlap: Probability in ``[0, 1]`` that a new ring reuses (shares)
            one account drawn from previously generated rings, producing
            overlapping rings instead of fully disjoint ones.
        decoy_ratio: Number of decoy LEGITIMATE high-value cycles to inject,
            as a fraction of *num_rings*.  Decoys are written into the normal
            transaction stream (``transactions/transactions_decoy.csv``) and are
            NOT recorded in ``fraud_cases.csv``.  They defeat pure
            amount-thresholding (they look high-value) and pure cycle-topology
            (they are cycles) without being fraud.
    """

    num_rings: int = 100
    depth_range: tuple[int, int] = (4, 7)
    amount: float = 9999.00
    amount_jitter: float = 0.0
    ring_overlap: float = 0.0
    decoy_ratio: float = 0.0
    _descriptions: list[str] = field(default_factory=lambda: SUSPICIOUS_DESCRIPTIONS)
    _decoy_descriptions: list[str] = field(default_factory=lambda: DECOY_DESCRIPTIONS)

    def _jittered_amount(self) -> float:
        """Return the fraud amount with optional relative jitter applied."""
        if self.amount_jitter <= 0.0:
            return self.amount
        low = self.amount * (1.0 - self.amount_jitter)
        high = self.amount * (1.0 + self.amount_jitter)
        return round(random.uniform(low, high), 2)

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
        # ids, then give each ring its own slice. Disjoint slices are the
        # default: overlapping ranges would merge two rings into a single
        # non-cycle component and make the per-ring involved_accounts labels
        # ambiguous. Deliberate overlap is opt-in via ``ring_overlap`` below.
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
        used_accounts: list[int] = []  # earlier-ring ids for optional overlap

        for pattern_id in tqdm(range(self.num_rings), desc="Generating fraud rings"):
            depth = depths[pattern_id]
            ring_ids = account_pool[pool_offset : pool_offset + depth]
            pool_offset += depth

            # Start from this ring's own disjoint slice of the account pool.
            account_ids = list(ring_ids)

            # At elevated hardness (ring_overlap > 0) deliberately merge with an
            # earlier ring by sharing one account. This is opt-in and makes the
            # detection problem harder; at ring_overlap == 0 every ring stays
            # fully disjoint, preserving the clean cycle labels.
            if used_accounts and random.random() < self.ring_overlap:
                shared = random.choice(used_accounts)
                if shared not in account_ids:
                    account_ids[random.randrange(depth)] = shared

            accounts = [f"acc_{a}" for a in account_ids]
            used_accounts.extend(account_ids)
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
                row.extend([self._jittered_amount(), "2024-01-01T12:00:00", desc])
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

        # Decoy legitimate high-value cycles go into the normal transaction
        # stream, never into fraud_cases.csv.
        num_decoys = int(round(self.num_rings * self.decoy_ratio))
        if num_decoys > 0:
            current_tx_id = self._generate_decoys(
                num_decoys=num_decoys,
                max_account_id=max_account_id,
                start_tx_id=current_tx_id,
                embedder=embedder,
                output_dir=output_dir,
                fmt=fmt,
                compress=compress,
            )

        return len(tx_rows), current_tx_id

    def _generate_decoys(
        self,
        num_decoys: int,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> int:
        """Inject *num_decoys* legitimate high-value cycles as normal traffic.

        Decoys are written to ``transactions/transactions_decoy.csv`` so they sit
        alongside the legitimate transactions, never in the fraud labels.

        Returns:
            The next unused transaction id.
        """
        import os

        tx_dir = os.path.join(output_dir, "transactions")
        os.makedirs(tx_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        decoy_rows: list[list] = []
        current_tx_id = start_tx_id

        for _ in range(num_decoys):
            min_d, max_d = self.depth_range
            depth = random.randint(min_d, max_d)
            if max_account_id < depth + 1:
                start_node = 0
            else:
                start_node = random.randint(0, max_account_id - depth - 1)
            accounts = [f"acc_{start_node + d}" for d in range(depth)]

            batch_texts: list[str] = []
            batch_rows: list[list] = []
            for k in range(depth):
                src = accounts[k]
                dst = accounts[(k + 1) % depth]
                desc = random.choice(self._decoy_descriptions)
                batch_texts.append(desc)
                row: list = [f"tx_{current_tx_id}", src, dst]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([self._jittered_amount(), "2024-01-01T13:00:00", desc])
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)
            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    decoy_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    decoy_rows.append(r + ["|".join(map(str, vec))])

        file_decoy = os.path.join(tx_dir, "transactions_decoy")
        write_output(file_decoy, headers_tx, decoy_rows, compress=compress)
        return current_tx_id
