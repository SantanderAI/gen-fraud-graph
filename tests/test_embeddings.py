# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.embeddings (fake, local and openai providers)."""

from __future__ import annotations

import sys
import types

import pytest

from gen_fraud_graph.embeddings import EmbeddingGenerator


class TestEmbeddings:
    def test_fake_provider_shape(self):
        emb = EmbeddingGenerator("fake", dim=128)
        result = emb.generate(["hello", "world"])
        assert result.shape == (2, 128)

    def test_fake_provider_empty(self):
        emb = EmbeddingGenerator("fake")
        result = emb.generate([])
        assert result == []

    def test_fake_provider_deterministic_shape(self):
        emb = EmbeddingGenerator("fake", dim=768)
        texts = [f"text_{i}" for i in range(100)]
        result = emb.generate(texts)
        assert result.shape == (100, 768)


class TestEmbeddingsProviders:
    def test_openai_missing_api_key(self, monkeypatch):
        fake_openai = types.ModuleType("openai")

        class FakeOpenAI:
            def __init__(self, api_key=None):
                self.api_key = api_key

        fake_openai.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            EmbeddingGenerator("openai", dim=8)

    def test_openai_generate(self, monkeypatch):
        fake_openai = types.ModuleType("openai")

        class _Data:
            def __init__(self, e):
                self.embedding = e

        class _Resp:
            def __init__(self, embs):
                self.data = [_Data(e) for e in embs]

        class _Embeddings:
            def create(self, input, model, dimensions):
                return _Resp([[0.1] * dimensions for _ in input])

        class _Client:
            def __init__(self, api_key=None):
                self.embeddings = _Embeddings()

        fake_openai.OpenAI = _Client
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        emb = EmbeddingGenerator("openai", dim=4)
        result = emb.generate(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_openai_import_error(self, monkeypatch):
        # Setting sys.modules[x] = None makes `import x` raise ImportError.
        monkeypatch.setitem(sys.modules, "openai", None)
        with pytest.raises(ImportError, match="openai is not installed"):
            EmbeddingGenerator("openai")

    def test_local_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="sentence-transformers is not installed"):
            EmbeddingGenerator("local")

    def test_local_generate(self, monkeypatch):
        import numpy as np

        fake_st = types.ModuleType("sentence_transformers")

        class _ST:
            def __init__(self, name):
                self.name = name

            def encode(self, texts):
                return np.zeros((len(texts), 4))

        fake_st.SentenceTransformer = _ST
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
        emb = EmbeddingGenerator("local")
        result = emb.generate(["a", "b"])
        assert result.shape == (2, 4)
