# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the synthetic fraud graph generator."""

from dataclasses import dataclass, field
from typing import Literal

Hardness = Literal["low", "medium", "high"]

# Difficulty presets controlling how hard the data is to separate by trivial
# heuristics.  ``low`` reproduces the original behaviour exactly (a single
# sentinel fraud amount, disjoint rings, no decoy cycles), so defaults are
# backward compatible.
_HARDNESS_PRESETS: dict[str, dict[str, float]] = {
    # amount_jitter:   relative +/- jitter applied to the fraud sentinel amount
    #                  (0.0 means every fraud edge keeps the exact sentinel).
    # ring_overlap:    probability that a new ring reuses (shares) an account
    #                  from an existing ring, creating overlapping rings.
    # decoy_ratio:     number of decoy legitimate high-value cycles to inject,
    #                  expressed as a fraction of the fraud-ring count.
    "low": {"amount_jitter": 0.0, "ring_overlap": 0.0, "decoy_ratio": 0.0},
    "medium": {"amount_jitter": 0.25, "ring_overlap": 0.25, "decoy_ratio": 0.5},
    "high": {"amount_jitter": 0.5, "ring_overlap": 0.5, "decoy_ratio": 1.0},
}


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
        hardness: Difficulty preset (``"low"``, ``"medium"`` or ``"high"``)
            controlling how hard the fraud data is to separate by trivial
            heuristics.  ``"low"`` (the default) is backward compatible: fraud
            edges keep a single sentinel amount, rings are disjoint, and no
            decoy cycles are injected.  Higher levels jitter fraud amounts,
            overlap rings, and inject legitimate high-value (decoy) cycles so
            that pure amount-thresholding and pure cycle-topology each fail.
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
    hardness: Hardness = "low"

    # Derived — computed in __post_init__
    num_accounts: int = field(init=False)
    num_transactions: int = field(init=False)
    amount_jitter: float = field(init=False)
    ring_overlap: float = field(init=False)
    decoy_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        self.num_accounts = int(10_000_000 * self.scale_factor)
        self.num_transactions = int(90_000_000 * self.scale_factor)
        if self.num_fraud_rings is None:
            self.num_fraud_rings = max(10, int(1000 * self.scale_factor))

        if self.hardness not in _HARDNESS_PRESETS:
            raise ValueError(
                f"hardness must be one of {sorted(_HARDNESS_PRESETS)}, got {self.hardness!r}"
            )
        preset = _HARDNESS_PRESETS[self.hardness]
        self.amount_jitter = preset["amount_jitter"]
        self.ring_overlap = preset["ring_overlap"]
        self.decoy_ratio = preset["decoy_ratio"]
