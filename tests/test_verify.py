# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.verify (pattern validation and its CLI)."""

from __future__ import annotations

import csv
import os
import sys

import pytest

from gen_fraud_graph.generator import FraudGraphGenerator
from gen_fraud_graph.verify import main as verify_main
from gen_fraud_graph.verify import verify_fraud_patterns


class TestVerify:
    def test_verify_valid_patterns(self, small_config):
        gen = FraudGraphGenerator(small_config)
        gen.run()

        cases_path = os.path.join(small_config.output_dir, "fraud", "fraud_cases.csv")
        assert verify_fraud_patterns(cases_path, small_config.output_dir)


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

    def test_cli_main_success(self, small_config, monkeypatch):
        FraudGraphGenerator(small_config).run()
        monkeypatch.setattr(sys, "argv", ["verify", "--data-dir", small_config.output_dir])
        with pytest.raises(SystemExit) as exc:
            verify_main()
        assert exc.value.code == 0
