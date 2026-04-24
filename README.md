# paper-embedder

Shared paper-embedding extraction library used by `zotero-mcp` and `My-Research-Assistant`.

> **Upgrading from 0.2.x → 0.3.0 (breaking change)**
>
> `PaperDescriptor.pdf_path: Path | None` has been replaced by `PaperDescriptor.fulltext_text: str | None`. The library no longer parses PDFs — callers supply extracted text directly. For the previous behavior (PDF → intro+methods markdown), install the `[marker]` extra and call `paper_embedder.markdown.prepare_fulltext(pdf_path, cache_path=...)` before constructing the descriptor. There is no backward-compatibility shim. `pdfminer.six` is no longer a transitive dependency of paper-embedder — consumers that still need it for other purposes must declare it explicitly.

## Install

Core (pure text-in, vectors-out):

```bash
pip install -e ".[dev]"
```

With PDF → markdown extraction (adds ~4–6 GB of Marker model weights on first use):

```bash
pip install -e ".[dev,marker]"
# macOS: brew install ghostscript      (optional, enables oversize-PDF compression)
# Linux: apt install ghostscript
```

## Usage

```python
import os
from pathlib import Path

from paper_embedder import (
    PaperDescriptor, ProviderConfig, get_provider, embed_paper, embed_query,
)

# Optional: PDF → intro+methods markdown via the [marker] extra.
from paper_embedder.markdown import prepare_fulltext

provider = get_provider(ProviderConfig(
    provider="gemini_v2",
    model_name="gemini-embedding-2-preview",
    api_key=os.environ["GEMINI_API_KEY"],
    dimension=1536,
))

# Index a paper — caller extracts text first, then passes it to embed_paper.
pdf = Path("/path/to/1706.03762.pdf")
cache = Path("/path/to/cache/attention_is_all_you_need.md")
fulltext_text = prepare_fulltext(pdf, cache_path=cache)  # str | None

paper = PaperDescriptor(
    paper_id="cvpr2024_123",
    title="Attention Is All You Need",
    abstract="We propose the Transformer, a new architecture...",
    fulltext_text=fulltext_text,       # None if extraction failed
    item_type="paper",
)
result = embed_paper(paper, provider)
print(result.abstract_vec.shape)       # (1536,)
if result.fulltext_vec is not None:
    print(result.fulltext_vec.shape)   # (1536,)
print(result.metadata["fingerprint"])

# Non-paper items (books / webpages / notes) — no PDF extraction.
non_paper = PaperDescriptor(
    paper_id="note_42",
    title="My Reading Notes",
    item_type="non_paper",
    creators="Self",
    notes="Three key takeaways...",
    tags="ml, reading",
)
result = embed_paper(non_paper, provider)  # abstract_vec only

# Embed a search query
query_vec = embed_query("What is attention in neural networks?", provider)
```

## Invariants

- **Read-only on inputs.** `embed_paper` does no filesystem I/O. `prepare_fulltext` reads the source PDF and writes only to the caller-provided `cache_path`.
- **No background state.** `embed_paper` is pure text-in, vectors-out.
- **Opt-in PDF pipeline.** `paper_embedder.markdown` is gated behind the `[marker]` extra. Importing it without `marker-pdf` installed raises `ImportError` with an install hint.
- **Section-gated extraction.** `prepare_fulltext` extracts only intro+methods (stops at the first experiments/results/conclusion heading). Callers never get back full-document markdown — this is deliberate, see `docs/superpowers/specs/2026-04-23-marker-hardening-design.md`.

## Development

```bash
pytest                                       # fast tests
MARKER_INTEGRATION=1 pytest -m slow          # slow integration tests (real Marker, real PDFs)
ruff check .
mypy paper_embedder
```
