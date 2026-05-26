# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""gen_fraud_graph — Synthetic fraud graph generator for financial crime detection research."""

__version__ = "0.1.0"

from gen_fraud_graph.config import Config
from gen_fraud_graph.generator import FraudGraphGenerator

__all__ = ["Config", "FraudGraphGenerator", "__version__"]
