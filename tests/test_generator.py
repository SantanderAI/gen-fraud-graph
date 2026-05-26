# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the synthetic fraud graph generator."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile

import pytest

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import get_headers, write_output
from gen_fraud_graph.generator import FraudGraphGenerator
from gen_fraud_graph.typologies import FraudRingGenerator
from gen_fraud_graph.verify import verify_fraud_patterns


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="gen_fraud_graph_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def small_config(tmp_dir):
    """A tiny config suitable for fast unit tests."""
    return Config(
        scale_factor=0.0001,
        embedding_provider="fake",
        workers=1,
        batches_per_worker=1,
        output_dir=tmp_dir,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert cfg.num_accounts == 10_000_000
        assert cfg.num_transactions == 90_000_000
        assert cfg.num_fraud_rings == 1000

    def test_scale_factor(self):
        cfg = Config(scale_factor=0.01)
        assert cfg.num_accounts == 100_000
        assert cfg.num_transactions == 900_000
        assert cfg.num_fraud_rings == max(10, int(1000 * 0.01))

    def test_explicit_fraud_rings(self):
        cfg = Config(num_fraud_rings=42)
        assert cfg.num_fraud_rings == 42

    def test_tiny_scale(self):
        cfg = Config(scale_factor=0.0001)
        assert cfg.num_accounts == 1_000
        assert cfg.num_transactions == 9_000
        assert cfg.num_fraud_rings >= 10


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_fake_provider_shape(self):
        emb = EmbeddingGenerator("fake", dim=128)
        result = emb.generate(["hello", "world"])
        assert result.shape == (2, 128)

    def test_fake_provider_empty(self):
        emb = EmbeddingGenerator("fake")
        result = emb.generate([])
        assert result == []

    def test_fake_provider_deterministic_shape(self):
        emb = EmbeddingGenerator("fake", dim=768)
        texts = [f"text_{i}" for i in range(100)]
        result = emb.generate(texts)
        assert result.shape == (100, 768)


# ---------------------------------------------------------------------------
# Exporter tests
# ---------------------------------------------------------------------------

class TestExporters:
    def test_csv_headers_account(self):
        h = get_headers("account", "csv")
        assert "account_id" in h
        assert "balance" in h

    def test_csv_headers_transaction(self):
        h = get_headers("transaction", "csv")
        assert "tx_id" in h
        assert "src_id" in h
        assert "dst_id" in h

    def test_neptune_headers_account(self):
        h = get_headers("account", "neptune")
        assert "~id" in h
        assert "~label" in h

    def test_neptune_headers_transaction(self):
        h = get_headers("transaction", "neptune")
        assert "~from" in h
        assert "~to" in h

    def test_write_output_csv(self, tmp_dir):
        path = os.path.join(tmp_dir, "test")
        write_output(path, ["a", "b"], [[1, 2], [3, 4]])
        assert os.path.exists(f"{path}.csv")

        with open(f"{path}.csv") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
        assert rows[0] == ["a", "b"]
        assert len(rows) == 3

    def test_write_output_compressed(self, tmp_dir):
        path = os.path.join(tmp_dir, "test_zip")
        write_output(path, ["x"], [[1], [2]], compress=True)
        assert os.path.exists(f"{path}.csv.zip")


# ---------------------------------------------------------------------------
# Fraud typology tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# End-to-end generator tests
# ---------------------------------------------------------------------------

class TestFraudGraphGenerator:
    def test_full_pipeline(self, small_config):
        gen = FraudGraphGenerator(small_config)
        gen.run()

        out = small_config.output_dir
        assert os.path.isdir(os.path.join(out, "accounts"))
        assert os.path.isdir(os.path.join(out, "transactions"))
        assert os.path.isdir(os.path.join(out, "fraud"))

        # Check that files are non-empty
        acc_files = os.listdir(os.path.join(out, "accounts"))
        assert len(acc_files) >= 1
        tx_files = os.listdir(os.path.join(out, "transactions"))
        assert len(tx_files) >= 1

    def test_skip_accounts(self, small_config):
        gen = FraudGraphGenerator(small_config)
        gen.run(skip_accounts=True)

        out = small_config.output_dir
        # accounts dir should not exist since we skipped
        assert not os.path.isdir(os.path.join(out, "accounts"))
        assert os.path.isdir(os.path.join(out, "transactions"))
        assert os.path.isdir(os.path.join(out, "fraud"))


# ---------------------------------------------------------------------------
# Verify tests
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_valid_patterns(self, small_config):
        gen = FraudGraphGenerator(small_config)
        gen.run()

        cases_path = os.path.join(small_config.output_dir, "fraud", "fraud_cases.csv")
        assert verify_fraud_patterns(cases_path, small_config.output_dir)
