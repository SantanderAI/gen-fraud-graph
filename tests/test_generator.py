# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.generator (workload planning, workers, pipeline)."""

from __future__ import annotations

import csv
import os
import shutil
import tempfile

import pytest

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import get_headers, write_output
from gen_fraud_graph.generator import (
    FraudGraphGenerator,
    _generate_accounts_chunk,
    _generate_transactions_chunk,
    _split_workload,
)
from gen_fraud_graph.typologies import FraudRingGenerator, StructuringGenerator
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

class TestWorkloadPlanning:
    def test_split_workload_distributes_remainder(self):
        shards = _split_workload(10, 3)
        assert shards == [(0, 4), (4, 3), (7, 3)]
        assert sum(count for _, count in shards) == 10

    def test_split_workload_handles_exact_division(self):
        shards = _split_workload(12, 4)
        assert shards == [(0, 3), (3, 3), (6, 3), (9, 3)]


class TestGeneratorWorkers:
    def test_accounts_chunk_csv(self, tmp_dir):
        msg = _generate_accounts_chunk(0, 0, 0, 20, "fake", 16, tmp_dir, "csv")
        assert "Generated" in msg
        assert os.path.exists(os.path.join(tmp_dir, "accounts", "accounts_0_0.csv"))

    def test_accounts_chunk_neptune(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 10, "fake", 16, tmp_dir, "neptune")
        path = os.path.join(tmp_dir, "accounts", "accounts_0_0.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~id" in header

    def test_accounts_chunk_resume_complete(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 5, "fake", 16, tmp_dir, "csv")
        msg = _generate_accounts_chunk(0, 0, 0, 5, "fake", 16, tmp_dir, "csv")
        assert "Skipped" in msg

    def test_accounts_chunk_resume_partial(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 5, "fake", 16, tmp_dir, "csv")
        msg = _generate_accounts_chunk(0, 0, 0, 10, "fake", 16, tmp_dir, "csv")
        assert "Generated" in msg
        path = os.path.join(tmp_dir, "accounts", "accounts_0_0.csv")
        with open(path) as fh:
            rows = list(csv.reader(fh))
        # header + 10 data rows
        assert len(rows) == 11

    def test_transactions_chunk_csv(self, tmp_dir):
        msg = _generate_transactions_chunk(0, 0, 0, 20, 100, "fake", 16, tmp_dir, "csv")
        assert "Generated" in msg
        assert os.path.exists(os.path.join(tmp_dir, "transactions", "transactions_0_0.csv"))

    def test_transactions_chunk_neptune(self, tmp_dir):
        _generate_transactions_chunk(0, 0, 0, 20, 100, "fake", 16, tmp_dir, "neptune")
        path = os.path.join(tmp_dir, "transactions", "transactions_0_0.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_transactions_chunk_resume_complete(self, tmp_dir):
        _generate_transactions_chunk(0, 0, 0, 5, 50, "fake", 16, tmp_dir, "csv")
        msg = _generate_transactions_chunk(0, 0, 0, 5, 50, "fake", 16, tmp_dir, "csv")
        assert "Skipped" in msg

    def test_transactions_chunk_resume_partial(self, tmp_dir):
        _generate_transactions_chunk(0, 0, 0, 5, 50, "fake", 16, tmp_dir, "csv")
        msg = _generate_transactions_chunk(0, 0, 0, 10, 50, "fake", 16, tmp_dir, "csv")
        assert "Generated" in msg


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


# ---------------------------------------------------------------------------
# Structuring typology test
# ---------------------------------------------------------------------------


class TestStructuringGenerator:
    def test_generate_creates_files(self, tmp_dir):
        """StructuringGenerator must write transactions_fraud.csv and fraud_cases.csv."""
        emb = EmbeddingGenerator("fake", dim=32)
        gen = StructuringGenerator(num_patterns=5, smurfs_range=(3, 3))

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

    def test_pattern_type_is_structuring(self, tmp_dir):
        """fraud_cases.csv rows from StructuringGenerator must have pattern_type='structuring'."""
        emb = EmbeddingGenerator("fake", dim=32)
        gen = StructuringGenerator(num_patterns=4, smurfs_range=(3, 3))
        gen.generate(
            max_account_id=200,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 4
        assert all(r["pattern_type"] == "structuring" for r in rows)

    def test_transaction_count_matches_smurfs(self, tmp_dir):
        """With fixed smurf count each pattern must emit exactly that many transactions."""
        emb = EmbeddingGenerator("fake", dim=32)
        fixed_smurfs = 5
        num_patterns = 3
        gen = StructuringGenerator(
            num_patterns=num_patterns,
            smurfs_range=(fixed_smurfs, fixed_smurfs),
        )
        n_tx, _ = gen.generate(
            max_account_id=500,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert n_tx == num_patterns * fixed_smurfs

    def test_amounts_are_sub_threshold(self, tmp_dir):
        """All structuring amounts must stay below the $10 000 CTR threshold."""
        emb = EmbeddingGenerator("fake", dim=32)
        gen = StructuringGenerator(num_patterns=10, smurfs_range=(3, 7))
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            reader = csv.DictReader(fh)
            amounts = [float(r["amount"]) for r in reader]
        assert all(a < 10_000.00 for a in amounts), "Found amount >= CTR threshold"

    def test_all_transactions_fan_into_coordinator(self, tmp_dir):
        """Every transaction in a structuring pattern must target the coordinator (fan-in star)."""
        emb = EmbeddingGenerator("fake", dim=32)
        fixed_smurfs = 4
        gen = StructuringGenerator(num_patterns=2, smurfs_range=(fixed_smurfs, fixed_smurfs))
        gen.generate(
            max_account_id=200,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        # Build a map of coordinator → smurf accounts from fraud_cases.csv
        coordinators: dict[str, set[str]] = {}
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            for row in csv.DictReader(fh):
                if row["pattern_type"] != "structuring":
                    continue
                coord = row["start_acc_id"]
                involved = set(row["involved_accounts"].split("|"))
                coordinators[coord] = involved

        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            for row in csv.DictReader(fh):
                dst = row["dst_id"]
                if dst in coordinators:
                    src = row["src_id"]
                    # source must be a known smurf for this coordinator
                    assert src in coordinators[dst], (
                        f"src {src} not a registered smurf of coordinator {dst}"
                    )

    def test_tx_ids_do_not_collide_with_start(self, tmp_dir):
        """Transaction IDs must begin at start_tx_id and never reuse prior IDs."""
        emb = EmbeddingGenerator("fake", dim=32)
        start = 9_999
        gen = StructuringGenerator(num_patterns=3, smurfs_range=(3, 3))
        _, next_id = gen.generate(
            max_account_id=500,
            start_tx_id=start,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            ids = [int(r["tx_id"].lstrip("tx_")) for r in csv.DictReader(fh)]
        assert min(ids) == start
        assert next_id == start + len(ids)

    def test_neptune_format(self, tmp_dir):
        """Neptune format must not include an embedding column."""
        emb = EmbeddingGenerator("fake", dim=32)
        gen = StructuringGenerator(num_patterns=2, smurfs_range=(3, 3))
        n_tx, _ = gen.generate(
            max_account_id=200,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            headers = next(csv.reader(fh))
        assert "embedding" not in headers

    def test_tiny_account_pool(self, tmp_dir):
        """Generator must not crash when max_account_id is smaller than smurfs_range[1] + 1."""
        emb = EmbeddingGenerator("fake", dim=32)
        gen = StructuringGenerator(num_patterns=5, smurfs_range=(3, 10))
        n_tx, _ = gen.generate(
            max_account_id=5,  # smaller than max smurfs + 1
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert n_tx > 0
