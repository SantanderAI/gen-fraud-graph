# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the test suite."""

from __future__ import annotations

import shutil
import tempfile

import pytest

from gen_fraud_graph.config import Config


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
