# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.cli."""

from __future__ import annotations

import csv
import os


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

    def test_main_falkordb_format(self, tmp_dir):
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
                "falkordb",
                "--fraud-rings",
                "3",
            ]
        )
        tx_path = os.path.join(tmp_dir, "transactions", "transactions_0_0.csv")
        with open(tx_path) as fh:
            header = next(csv.reader(fh))
        assert header[0] == ":START_ID(Account)"
        assert header[1] == ":END_ID(Account)"
        assert header[-1] == "is_fraud:BOOLEAN"
