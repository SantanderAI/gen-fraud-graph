# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Additional tests targeting under-covered modules (CLI, workers, providers)."""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
import types

import pytest

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import append_csv
from gen_fraud_graph.generator import (
    FraudGraphGenerator,
    _generate_accounts_chunk,
    _generate_transactions_chunk,
)
from gen_fraud_graph.typologies import FraudRingGenerator
from gen_fraud_graph.verify import main as verify_main
from gen_fraud_graph.verify import verify_fraud_patterns


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp(prefix="gen_fraud_graph_cov_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_full(self, tmp_dir):
        from gen_fraud_graph.cli import main

        main(
            [
                "--scale",
                "0.0001",
                "--provider",
                "fake",
                "--output",
                tmp_dir,
                "--workers",
                "1",
                "--batches",
                "1",
                "--format",
                "csv",
                "--fraud-rings",
                "5",
            ]
        )
        assert os.path.isdir(os.path.join(tmp_dir, "accounts"))
        assert os.path.isdir(os.path.join(tmp_dir, "fraud"))

    def test_main_skip_accounts_and_compress(self, tmp_dir):
        from gen_fraud_graph.cli import main

        main(
            [
                "--scale",
                "0.0001",
                "--output",
                tmp_dir,
                "--fraud-rings",
                "3",
                "--skip-accounts",
                "--compress",
            ]
        )
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv.zip"))


# ---------------------------------------------------------------------------
# Exporters — append_csv
# ---------------------------------------------------------------------------


class TestAppendCsv:
    def test_append_creates_file_with_header(self, tmp_dir):
        path = os.path.join(tmp_dir, "x.csv")
        append_csv(path, ["a", "b"], [[1, 2]])
        with open(path) as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["a", "b"]
        assert rows[1] == ["1", "2"]

    def test_append_existing_file_skips_header(self, tmp_dir):
        path = os.path.join(tmp_dir, "x.csv")
        append_csv(path, ["a", "b"], [[1, 2]])
        append_csv(path, ["a", "b"], [[3, 4]])
        with open(path) as fh:
            rows = list(csv.reader(fh))
        assert rows == [["a", "b"], ["1", "2"], ["3", "4"]]


# ---------------------------------------------------------------------------
# Generator worker functions (direct, single-process for coverage)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Embeddings — local/openai providers (mocked)
# ---------------------------------------------------------------------------


class TestEmbeddingsProviders:
    def test_openai_missing_api_key(self, monkeypatch):
        fake_openai = types.ModuleType("openai")

        class FakeOpenAI:
            def __init__(self, api_key=None):
                self.api_key = api_key

        fake_openai.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            EmbeddingGenerator("openai", dim=8)

    def test_openai_generate(self, monkeypatch):
        fake_openai = types.ModuleType("openai")

        class _Data:
            def __init__(self, e):
                self.embedding = e

        class _Resp:
            def __init__(self, embs):
                self.data = [_Data(e) for e in embs]

        class _Embeddings:
            def create(self, input, model, dimensions):
                return _Resp([[0.1] * dimensions for _ in input])

        class _Client:
            def __init__(self, api_key=None):
                self.embeddings = _Embeddings()

        fake_openai.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        emb = EmbeddingGenerator("openai", dim=4)
        result = emb.generate(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_openai_import_error(self, monkeypatch):
        # Setting sys.modules[x] = None makes `import x` raise ImportError.
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="openai is not installed"):
            EmbeddingGenerator("openai")

    def test_local_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="sentence-transformers is not installed"):
            EmbeddingGenerator("local")

    def test_local_generate(self, monkeypatch):
        import numpy as np

        fake_st = types.ModuleType("sentence_transformers")

        class _ST:
            def __init__(self, name):
                self.name = name

            def encode(self, texts):
                return np.zeros((len(texts), 4))

        fake_st.SentenceTransformer = _ST
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
        emb = EmbeddingGenerator("local")
        result = emb.generate(["a", "b"])
        assert result.shape == (2, 4)


# ---------------------------------------------------------------------------
# Verify — error paths & CLI
# ---------------------------------------------------------------------------


class TestVerifyExtra:
    def test_missing_fraud_tx_file(self, tmp_dir, capsys):
        cases = os.path.join(tmp_dir, "fraud_cases.csv")
        with open(cases, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["pattern_id", "start_acc_id", "pattern_type", "depth", "involved_accounts"]
            )
            writer.writerow(["pat_0", "acc_0", "cycle", 3, "acc_0|acc_1|acc_2"])
        assert verify_fraud_patterns(cases, tmp_dir) is False
        assert "not found" in capsys.readouterr().err

    def test_invalid_pattern_detected(self, tmp_dir):
        tx = os.path.join(tmp_dir, "transactions_fraud.csv")
        cases = os.path.join(tmp_dir, "fraud_cases.csv")
        with open(tx, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["tx_id", "src_id", "dst_id", "amount", "timestamp", "description", "embedding"]
            )
            # Only one edge present; cycle of 3 requires three edges → must fail
            w.writerow(["tx_0", "acc_0", "acc_1", 1.0, "2024", "x", ""])
        with open(cases, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pattern_id", "start_acc_id", "pattern_type", "depth", "involved_accounts"])
            w.writerow(["pat_0", "acc_0", "cycle", 3, "acc_0|acc_1|acc_2"])
        assert verify_fraud_patterns(cases, tmp_dir) is False

    def test_cli_main_missing_cases(self, tmp_dir, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["verify", "--data-dir", tmp_dir])
        with pytest.raises(SystemExit) as exc:
            verify_main()
        assert exc.value.code == 1

    def test_cli_main_success(self, tmp_dir, monkeypatch):
        cfg = Config(
            scale_factor=0.0001,
            embedding_provider="fake",
            workers=1,
            batches_per_worker=1,
            output_dir=tmp_dir,
        )
        FraudGraphGenerator(cfg).run()
        monkeypatch.setattr(sys, "argv", ["verify", "--data-dir", tmp_dir])
        with pytest.raises(SystemExit) as exc:
            verify_main()
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Typologies — neptune format, edge cases, compression
# ---------------------------------------------------------------------------


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

    def test_small_max_account_id(self, tmp_dir):
        """max_account_id < depth+1 must take the start_node=0 fallback."""
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(4, 4))
        n_tx, _ = gen.generate(
            max_account_id=3,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0

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
