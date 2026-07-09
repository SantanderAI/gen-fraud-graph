# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.typologies (fraud ring generation)."""

from __future__ import annotations

import csv
import os
import random

import pytest

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.typologies import FraudRingGenerator


class TestFraudRings:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = FraudRingGenerator(num_rings=5, depth_range=(3, 5))
        n_tx, next_id = gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))

    def test_fraud_cases_have_correct_columns(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = FraudRingGenerator(num_rings=3, depth_range=(4, 4))
        gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        assert "pattern_id" in rows[0]
        assert "involved_accounts" in rows[0]

    def test_rings_use_disjoint_accounts(self, tmp_dir):
        # Each ring must occupy its own accounts. Overlapping ranges merge two
        # rings into a single non-cycle component and make the per-ring
        # involved_accounts labels ambiguous.
        random.seed(0)
        emb = EmbeddingGenerator("fake", dim=8)
        gen = FraudRingGenerator(num_rings=15, depth_range=(4, 4))
        gen.generate(
            max_account_id=80,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        seen: set[str] = set()
        for row in rows:
            accounts = row["involved_accounts"].split("|")
            assert seen.isdisjoint(
                accounts
            ), f"{row['pattern_id']} reuses accounts from an earlier ring"
            seen.update(accounts)


class TestTypologiesExtra:
    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(3, 3))
        n_tx, _ = gen.generate(
            max_account_id=50,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_oversubscribed_rings_raise(self, tmp_dir):
        """When the rings need more distinct accounts than exist they can't be
        packed disjointly, so generate() must raise rather than emit rings that
        reference nonexistent accounts."""
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(4, 4))
        with pytest.raises(ValueError, match="distinct"):
            gen.generate(
                max_account_id=3,
                start_tx_id=0,
                embedder=emb,
                output_dir=tmp_dir,
                fmt="csv",
            )

    def test_compress(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(3, 3))
        gen.generate(
            max_account_id=50,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
            compress=True,
        )
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv.zip"))
