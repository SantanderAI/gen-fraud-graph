# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Embedding generators for node/edge feature vectors."""

from __future__ import annotations

import sys
from typing import Literal, Sequence

import numpy as np


class EmbeddingGenerator:
    """Generate dense vector embeddings for text descriptions.

    Supports three backends:

    * ``"fake"`` — fast random vectors (no extra dependencies).
    * ``"local"`` — ``sentence-transformers`` with ``all-MiniLM-L6-v2``.
    * ``"openai"`` — OpenAI ``text-embedding-3-small`` via API.

    Args:
        provider: One of ``"fake"``, ``"local"``, ``"openai"``.
        dim: Embedding dimensionality (used by *fake* and *openai*).
    """

    def __init__(
        self,
        provider: Literal["fake", "local", "openai"] = "fake",
        dim: int = 768,
    ) -> None:
        self.provider = provider
        self.dim = dim
        self._model = None
        self._client = None

        if provider == "local":
            try:
                from sentence_transformers import SentenceTransformer

                print("Loading local SentenceTransformer model...")
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                print(
                    "ERROR: sentence-transformers not installed.\n"
                    "Install with: pip install 'gen-fraud-graph[local]'\n"
                    "Or use --provider fake (random vectors).",
                    file=sys.stderr,
                )
                sys.exit(1)

        elif provider == "openai":
            import os

            try:
                from openai import OpenAI
            except ImportError:
                print(
                    "ERROR: openai package not installed.\n"
                    "Install with: pip install 'gen-fraud-graph[openai]'",
                    file=sys.stderr,
                )
                sys.exit(1)

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for the openai provider"
                )
            self._client = OpenAI(api_key=api_key)

    def generate(self, texts: Sequence[str]) -> list[list[float]] | np.ndarray:
        """Return embeddings for a batch of *texts*.

        Returns:
            For *fake*/*local*: a ``numpy.ndarray`` of shape ``(len(texts), dim)``.
            For *openai*: a list of float-lists.
        """
        if not texts:
            return []

        if self.provider == "fake":
            return np.random.rand(len(texts), self.dim).astype("float32")

        if self.provider == "local":
            return self._model.encode(texts)  # type: ignore[union-attr]

        if self.provider == "openai":
            return self._generate_openai(texts)

        return []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_openai(self, texts: Sequence[str]) -> list[list[float]]:
        """Call OpenAI embeddings API with automatic batching."""
        batch_size = 1000
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.embeddings.create(  # type: ignore[union-attr]
                input=list(batch),
                model="text-embedding-3-small",
                dimensions=self.dim,
            )
            all_embeddings.extend([d.embedding for d in response.data])

        return all_embeddings
