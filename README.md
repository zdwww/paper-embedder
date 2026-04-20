# paper-embedder

Shared paper-embedding extraction library used by `zotero-mcp` and `My-Research-Assistant`.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```python
from paper_embedder import (
    PaperDescriptor, ProviderConfig, get_provider, embed_paper, embed_query,
)

provider = get_provider(ProviderConfig(
    provider="gemini_v2",
    model_name="gemini-embedding-2-preview",
    api_key="...",
    dimension=1536,
))

result = embed_paper(
    PaperDescriptor(
        paper_id="cvpr2024_123",
        title="Title",
        abstract="Abstract text.",
        pdf_path=Path("/path/to/paper.pdf"),
        item_type="paper",
    ),
    provider,
)
# result.abstract_vec, result.fulltext_vec
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
