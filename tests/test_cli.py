# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.cli."""

from __future__ import annotations

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
