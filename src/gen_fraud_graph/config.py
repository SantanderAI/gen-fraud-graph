# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the synthetic fraud graph generator."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    """Generator configuration.

    Args:
        scale_factor: Multiplier over the base sizes.  ``1.0`` produces ~10 M
            accounts and ~90 M transactions.  Use ``0.01`` for ~100 K accounts.
        num_fraud_rings: Number of cyclic fraud patterns to inject.  When
            *None* it is derived automatically from *scale_factor*.
        fraud_ring_depth_range: Min/max depth (hops) of each fraud ring.
        embedding_provider: ``"fake"`` (random vectors, no deps), ``"local"``
            (SentenceTransformers), or ``"openai"`` (requires API key).
        embedding_dim: Dimensionality of generated embeddings.
        workers: Parallel processes for account/transaction generation.
        batches_per_worker: File chunks each worker produces.
        output_format: ``"csv"`` (generic) or ``"neptune"`` (AWS Neptune
            bulk-load headers).
        compress: Whether to ZIP the output CSV files.
        output_dir: Destination directory for generated files.
    """

    scale_factor: float = 1.0
    num_fraud_rings: int | None = None
    fraud_ring_depth_range: tuple[int, int] = (4, 7)
    embedding_provider: Literal["fake", "local", "openai"] = "fake"
    embedding_dim: int = 768
    workers: int = 1
    batches_per_worker: int = 1
    output_format: Literal["csv", "neptune"] = "csv"
    compress: bool = False
    output_dir: str = "data"

    # Derived — computed in __post_init__
    num_accounts: int = field(init=False)
    num_transactions: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_accounts = int(10_000_000 * self.scale_factor)
        self.num_transactions = int(90_000_000 * self.scale_factor)
        if self.num_fraud_rings is None:
            self.num_fraud_rings = max(10, int(1000 * self.scale_factor))
