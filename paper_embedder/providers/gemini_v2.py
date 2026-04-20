"""Gemini v2 preview embedding provider.

Gemini v2 (gemini-embedding-2-preview) does NOT use the task_type config field;
instead, we prepend a natural-language prefix to each input text to signal
document vs. query intent. Hard token cap is 8192; we reserve 20 tokens for the
prefix and truncate the body to fit the remaining ~7980 tokens using a
char-based estimate (≈4 chars/token).
"""

from __future__ import annotations

import hashlib
import time
from typing import Literal

import numpy as np
import numpy.typing as npt
from google import genai
from google.genai import types

from paper_embedder.errors import ProviderError

_V2_DOC_PREFIX = "Represent this document for retrieval: "
_V2_QUERY_PREFIX = "Represent this query for retrieval: "
_V2_HARD_CAP_TOKENS = 8192
_V2_PREFIX_TOKEN_BUDGET = 20
_V2_MAX_INPUT_TOKENS = _V2_HARD_CAP_TOKENS - _V2_PREFIX_TOKEN_BUDGET  # 8172
_CHARS_PER_TOKEN_ESTIMATE = 4
# Effective safe budget in chars, tuned one token below ceiling to stay comfortably under cap.
_V2_MAX_INPUT_CHARS = (_V2_MAX_INPUT_TOKENS - 192) * _CHARS_PER_TOKEN_ESTIMATE  # ≈ 31920


_TRANSIENT_MARKERS = ("RateLimit", "Server", "Internal", "Unavailable", "Deadline")
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (1.0, 4.0, 16.0)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    return any(marker in name for marker in _TRANSIENT_MARKERS)


class GeminiV2Provider:
    """Embedding provider for Google Gemini v2 preview models."""

    name: str = "gemini_v2"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        dim: int,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self.dim = dim
        self._base_url = base_url
        self.max_input_tokens = _V2_MAX_INPUT_TOKENS - 192  # keep consistent with char budget
        self._client: genai.Client | None = None  # lazy — instantiated on first embed() call

    def fingerprint(self) -> str:
        payload = f"{self.name}|{self._model_name}|{self.dim}|v2-prefix-injection"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def truncate(self, text: str) -> str:
        if len(text) <= _V2_MAX_INPUT_CHARS:
            return text
        return text[:_V2_MAX_INPUT_CHARS]

    def embed(
        self,
        texts: list[str],
        *,
        mode: Literal["document", "query"],
    ) -> list[npt.NDArray[np.float32]]:
        prefix = _V2_DOC_PREFIX if mode == "document" else _V2_QUERY_PREFIX
        prefixed = [prefix + self.truncate(t) for t in texts]

        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)

        attempt = 0
        while True:
            try:
                response = self._client.models.embed_content(
                    model=self._model_name,
                    contents=prefixed,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dim,
                    ),
                )
                embeddings = response.embeddings or []
                return [
                    np.asarray(emb.values, dtype=np.float32)
                    for emb in embeddings
                ]
            except Exception as exc:
                if not _is_transient(exc) or attempt >= _MAX_RETRIES:
                    raise ProviderError(
                        f"Gemini v2 embed failed ({type(exc).__name__}): {exc}"
                    ) from exc
                time.sleep(_BACKOFF_SECONDS[attempt])
                attempt += 1
