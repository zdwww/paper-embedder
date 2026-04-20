# paper-embedder

Shared paper-embedding extraction library used by `zotero-mcp` and `My-Research-Assistant`.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
import os
from pathlib import Path

from paper_embedder import (
    PaperDescriptor, ProviderConfig, get_provider, embed_paper, embed_query,
)

provider = get_provider(ProviderConfig(
    provider="gemini_v2",
    model_name="gemini-embedding-2-preview",
    api_key=os.environ["GEMINI_API_KEY"],
    dimension=1536,
))

# Index a paper
paper = PaperDescriptor(
    paper_id="cvpr2024_123",
    title="Attention Is All You Need",
    abstract="We propose the Transformer, a new architecture...",
    pdf_path=Path("/path/to/1706.03762.pdf"),
    item_type="paper",
)
result = embed_paper(paper, provider)
print(result.abstract_vec.shape)    # (1536,)
print(result.fulltext_vec.shape)    # (1536,) — or None if extraction failed
print(result.metadata["fingerprint"])

# Index a non-paper (book / webpage / note)
non_paper = PaperDescriptor(
    paper_id="note_42",
    title="My Reading Notes",
    item_type="non_paper",
    creators="Self",
    notes="Three key takeaways...",
    tags="ml, reading",
)
result = embed_paper(non_paper, provider)
# result.abstract_vec only; fulltext_vec is None

# Embed a search query
query_vec = embed_query("What is attention in neural networks?", provider)
```

## Invariants

- **Read-only on inputs.** Never writes, renames, or modifies original PDFs or source metadata.
- **No filesystem writes.** All outputs are returned in-memory; callers own storage.

## Development

```bash
pytest
ruff check .
mypy paper_embedder
```
