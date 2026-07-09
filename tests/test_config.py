# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.config."""

from __future__ import annotations

from gen_fraud_graph.config import Config


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
