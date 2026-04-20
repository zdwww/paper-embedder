"""GeminiV2Provider — protocol compliance, fingerprint stability, truncation budget."""

from unittest.mock import MagicMock, patch

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


class _FakeEmbedResult:
    def __init__(self, vectors: list[list[float]]):
        self.embeddings = [MagicMock(values=v) for v in vectors]


def _make_fake_client(output_vectors_per_call: list[list[list[float]]]) -> MagicMock:
    """Returns a MagicMock genai.Client whose .models.embed_content returns
    one _FakeEmbedResult per call, in order."""
    client = MagicMock()
    results = [_FakeEmbedResult(vs) for vs in output_vectors_per_call]
    client.models.embed_content.side_effect = results
    return client


def test_embed_document_mode_prepends_doc_prefix():
    from paper_embedder.providers.gemini_v2 import _V2_DOC_PREFIX, GeminiV2Provider

    fake = _make_fake_client([[[0.1] * 1536, [0.2] * 1536]])
    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        vecs = p.embed(["doc A", "doc B"], mode="document")

    assert len(vecs) == 2
    assert vecs[0].shape == (1536,)
    call_kwargs = fake.models.embed_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert all(c.startswith(_V2_DOC_PREFIX) for c in contents)
    assert contents[0] == _V2_DOC_PREFIX + "doc A"


def test_embed_query_mode_prepends_query_prefix():
    from paper_embedder.providers.gemini_v2 import _V2_QUERY_PREFIX, GeminiV2Provider

    fake = _make_fake_client([[[0.3] * 1536]])
    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        p.embed(["what is attention?"], mode="query")

    contents = fake.models.embed_content.call_args.kwargs["contents"]
    assert contents[0] == _V2_QUERY_PREFIX + "what is attention?"


def test_embed_returns_float32_arrays():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    fake = _make_fake_client([[[0.1] * 1536]])
    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        vecs = p.embed(["hi"], mode="document")

    assert vecs[0].dtype == np.float32


def test_embed_passes_model_name():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    fake = _make_fake_client([[[0.1] * 1536]])
    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        p.embed(["hi"], mode="document")

    assert fake.models.embed_content.call_args.kwargs["model"] == "gemini-embedding-2-preview"


def test_embed_passes_output_dimensionality_to_api():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    fake = _make_fake_client([[[0.1] * 3072]])
    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=3072)
        p.embed(["hi"], mode="document")

    call_kwargs = fake.models.embed_content.call_args.kwargs
    assert "config" in call_kwargs
    assert call_kwargs["config"].output_dimensionality == 3072


def test_embed_passes_output_dimensionality_matches_dim_config():
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    fake = _make_fake_client([[[0.1] * 768]])
    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=768)
        p.embed(["hi"], mode="document")

    call_kwargs = fake.models.embed_content.call_args.kwargs
    assert call_kwargs["config"].output_dimensionality == 768


def test_embed_retries_on_transient_error_then_succeeds(monkeypatch):
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    # no-op sleep so the test is fast
    monkeypatch.setattr("paper_embedder.providers.gemini_v2.time.sleep", lambda _: None)

    class _Transient(Exception):
        pass
    _Transient.__name__ = "RateLimitError"

    fake = MagicMock()
    good_result = _FakeEmbedResult([[0.5] * 1536])
    fake.models.embed_content.side_effect = [_Transient("slow down"), good_result]

    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        vecs = p.embed(["hi"], mode="document")

    assert len(vecs) == 1
    assert fake.models.embed_content.call_count == 2


def test_embed_raises_provider_error_after_3_transient_failures(monkeypatch):
    from paper_embedder.errors import ProviderError
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    monkeypatch.setattr("paper_embedder.providers.gemini_v2.time.sleep", lambda _: None)

    class _Transient(Exception):
        pass
    _Transient.__name__ = "ServerError"

    fake = MagicMock()
    fake.models.embed_content.side_effect = _Transient("boom")

    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        with pytest.raises(ProviderError):
            p.embed(["hi"], mode="document")

    # initial call + 3 retries == 4 total attempts
    assert fake.models.embed_content.call_count == 4


def test_embed_permanent_error_raises_immediately_no_retry():
    from paper_embedder.errors import ProviderError
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    class _Permanent(Exception):
        pass
    _Permanent.__name__ = "InvalidArgumentError"

    fake = MagicMock()
    fake.models.embed_content.side_effect = _Permanent("bad input")

    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        with pytest.raises(ProviderError):
            p.embed(["hi"], mode="document")

    assert fake.models.embed_content.call_count == 1


def test_embed_uses_exponential_backoff_timings(monkeypatch):
    from paper_embedder.providers.gemini_v2 import GeminiV2Provider

    sleeps: list[float] = []
    monkeypatch.setattr(
        "paper_embedder.providers.gemini_v2.time.sleep",
        lambda s: sleeps.append(s),
    )

    class _Transient(Exception):
        pass
    _Transient.__name__ = "RateLimitError"

    fake = MagicMock()
    fake.models.embed_content.side_effect = [
        _Transient(), _Transient(), _FakeEmbedResult([[0.1] * 1536])
    ]

    with patch("paper_embedder.providers.gemini_v2.genai.Client", return_value=fake):
        p = GeminiV2Provider(api_key="k", model_name="gemini-embedding-2-preview", dim=1536)
        p.embed(["hi"], mode="document")

    assert sleeps == [1.0, 4.0]
