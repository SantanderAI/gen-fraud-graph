# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.generator (workload planning, workers, pipeline)."""

from __future__ import annotations

import csv
import os

from gen_fraud_graph.generator import (
    FraudGraphGenerator,
    _generate_accounts_chunk,
    _generate_transactions_chunk,
    _split_workload,
)


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
