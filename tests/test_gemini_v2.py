"""GeminiV2Provider — protocol compliance, fingerprint stability, truncation budget."""

import hashlib

import numpy as np
import pytest


def test_gemini_v2_constructor_sets_attrs():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
    assert p.name == "gemini_v2"
    assert p.dim == 1536
    assert p.max_input_tokens == 7980


def test_gemini_v2_is_protocol_compliant():
    from paper_embedder.providers.base import Provider
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
    assert isinstance(p, Provider)


def test_gemini_v2_fingerprint_is_stable():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p1 = GeminiV2Provider(api_key="k1", model_name="gemini-embedding-2-preview", dim=1536)
    p2 = GeminiV2Provider(api_key="k2", model_name="gemini-embedding-2-preview", dim=1536)
    assert p1.fingerprint() == p2.fingerprint(), "API key must not affect fingerprint"


def test_gemini_v2_fingerprint_differs_by_model():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p1 = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
    p2 = GeminiV2Provider(api_key="k", model_name="gemini-embedding-001", dim=1536)
    assert p1.fingerprint() != p2.fingerprint()


def test_gemini_v2_fingerprint_differs_by_dim():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p1 = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=768)
    p2 = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
    assert p1.fingerprint() != p2.fingerprint()


def test_gemini_v2_truncate_leaves_short_text_intact():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
    assert p.truncate("hello world") == "hello world"


def test_gemini_v2_truncate_shortens_long_text_below_budget():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
    long_text = "x" * 100_000
    truncated = p.truncate(long_text)
    # char budget ≈ 7980 tokens * 4 chars/token = 31920
    assert len(truncated) <= 31920
    assert len(truncated) < len(long_text)
