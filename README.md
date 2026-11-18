# RAAG: Relationship-Aware Augmented Generation

**Structure-derived document relationship intelligence for RAG pipelines.**

> **Project status: ended.** Development of RAAG has stopped and we have moved our focus to other work. The code stays available as is under the MIT license. No further features, fixes or support are planned.

RAAG is a Python library for storing and searching documents in a RAG pipeline, with tenant isolation built in. It provides the storage and search layer:

- Heading-aware Markdown splitting. Each section keeps its heading, section number and parent headings.
- SQLite storage for documents and sections, through SQLAlchemy (async).
- A `VectorStore` protocol, with an in-memory implementation and a Qdrant adapter.
- Tenant isolation on every operation, set with `tenant_context`.
- An async API, plus sync wrappers.

You bring the embedding model. RAAG splits, stores and searches.

Relationship extraction and graph-guided retrieval were planned and were not built. See [What was planned](#what-was-planned).

---

## Installation

```bash
pip install raag
```

Requires Python 3.10 or later. The core install has no vector database dependency and uses the built-in `InMemoryVectorStore`.

To use the Qdrant adapter, install the `qdrant` extra:

```bash
pip install "raag[qdrant]"
```

---

## Quick Start

```python
import asyncio
from raag import Raag, tenant_context


async def embed(texts: list[str]) -> list[list[float]]:
    # Stand-in for a real embedding model: one vector per input text.
    return [[float(t.lower().count(c)) for c in "aeiou"] for t in texts]


POLICY = """# Leave Policy

## 2.1 Annual Leave
Employees get 30 days of annual leave per year.

## 4.1 Maternity Leave
Maternity leave is 90 days.
"""


async def main():
    raag = Raag(embed_fn=embed, db_path="raag.db")

    async with tenant_context("acme"):
        doc_id = await raag.add_document(POLICY, {"title": "Leave Policy"})

        query = (await embed(["How much annual leave do I get?"]))[0]
        for r in await raag.search(query, k=3):
            print(f"{r.score:.3f}  {r.text.splitlines()[0]}")


asyncio.run(main())
```

`embed_fn` must be an async function that takes a list of strings and returns one vector per string. RAAG calls it when adding documents, and raises `ValueError` if it returns a different number of vectors. For search, you embed the query yourself and pass the vector.

### Sync usage

Every write and search method has a sync wrapper. `tenant_context` works with plain `with`:

```python
from raag import Raag, tenant_context


async def embed(texts: list[str]) -> list[list[float]]:
    return [[float(t.lower().count(c)) for c in "aeiou"] for t in texts]


raag = Raag(embed_fn=embed, db_path="raag.db")

with tenant_context("acme"):
    doc_id = raag.add_document_sync("# Notes\n\nSync usage works too.", {"title": "Notes"})
    results = raag.search_sync([1.0, 0.0, 1.0, 1.0, 1.0], k=5)
    raag.delete_document_sync(doc_id)
```

The sync wrappers also work when called from inside a running event loop (for example in Jupyter or an async web framework), and one instance can be shared by several threads. The current tenant is carried over in every case. Each instance runs its database work on its own background thread.

---

## Multi-Tenant Usage

Every operation requires a tenant. Set it with `tenant_context`. Calling any `Raag` method without one raises `RuntimeError`.

Documents and sections are stored with their `tenant_id`, and every read and delete is filtered by the current tenant, so one tenant never sees another tenant's data:

```python
import asyncio
from raag import Raag, tenant_context


async def embed(texts: list[str]) -> list[list[float]]:
    return [[float(t.lower().count(c)) for c in "aeiou"] for t in texts]


async def main():
    raag = Raag(embed_fn=embed, db_path=":memory:")  # nothing persists between runs

    async with tenant_context("client_a"):
        await raag.add_document("# A\n\nConfidential to client A.", {"title": "A"})

    async with tenant_context("client_b"):
        await raag.add_document("# B\n\nConfidential to client B.", {"title": "B"})
        docs = await raag.list_documents()
        print([d.title for d in docs])  # ['B']

        query = (await embed(["confidential"]))[0]
        results = await raag.search(query)
        print({r.doc_id for r in results} == {d.doc_id for d in docs})  # True


asyncio.run(main())
```

Use `get_current_tenant()` to read the current tenant (`None` if unset), or `require_tenant()` to read it and raise if unset.

---

## How Documents Are Split

The default splitter is `MarkdownSplitter`. Each Markdown heading (`#` to `######`) starts a new section, which runs until the next heading of any level.

- The heading line is kept at the top of the section text.
- A leading number in a heading, such as `## 2.1 Annual Leave`, is stored as the section number.
- Text before the first heading becomes its own section.
- Headings with no body text are not stored as sections. They still appear in the parent path of the sections below them.
- A section longer than `max_section_chars` is split on paragraph breaks. Each extra chunk repeats the heading with ` (continued)`.

Plain text with no headings is treated as a single section and split on paragraph breaks if it is too long.

The splitter metadata (`heading_level`, `section_number`, `section_title`, `parent_path`, `chunk_index`, `total_chunks`, plus `source_path` if you pass it in the document metadata) is stored with each section in SQLite. You can also run the splitter on its own:

```python
from raag import MarkdownSplitter

sections = MarkdownSplitter().split("# Policy\n\n## 2.1 Leave\nThirty days.", max_chars=1000)
for s in sections:
    print(s.section_number, s.section_title, s.parent_path)  # 2.1 Leave ['Policy']
```

To use a different splitter, pass any object with a `split(text: str, max_chars: int = 1000) -> list[SplitSection]` method as `splitter=`.

---

## API Reference

All names below are importable from `raag`, except `QdrantVectorStore`.

### `Raag`

```python
Raag(
    embed_fn,                 # async (list[str]) -> list[list[float]]
    db_path="raag.db",        # SQLite file path, ":memory:", or a full sqlite+aiosqlite:// URL
    vector_store=None,        # defaults to InMemoryVectorStore()
    config=None,              # defaults to RaagConfig()
    splitter=None,            # defaults to MarkdownSplitter()
)
```

Tables are created on first use. All methods require a tenant context. Only SQLite is supported: a URL for any other database raises `ValueError`, as does a value that starts like a URL scheme without `//` (such as `postgresql:...`). Start a file name that contains a colon with `./`.

| Method | Returns | Description |
|--------|---------|-------------|
| `await add_document(text, metadata)` | `str` | Splits, embeds and stores a document. Returns the document ID. `metadata` is a dict (it may be empty). Its `title` key sets the document title (default `"Untitled"`). A document with no text is stored with no sections. If storing fails, the document's rows and vectors are removed before the error is raised. |
| `await search(query_embedding, k=None)` | `list[SearchResult]` | Returns the `k` most similar sections, highest score first. `k` defaults to `config.default_k`. |
| `await delete_document(doc_id)` | `None` | Deletes a document, its sections and their vectors. |
| `await list_documents()` | `list[Document]` | Lists the current tenant's documents, oldest first. |
| `add_document_sync(text, metadata)` | `str` | Sync wrapper. |
| `search_sync(query_embedding, k=None)` | `list[SearchResult]` | Sync wrapper. |
| `delete_document_sync(doc_id)` | `None` | Sync wrapper. |

### `RaagConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `default_k` | `10` | Number of results `search` returns when `k` is not given. |
| `max_section_chars` | `1000` | Sections longer than this are split into chunks. |
| `embed_batch_size` | `100` | Maximum number of section texts sent to `embed_fn` in one call. Must be at least 1. |

### Models

- `SearchResult`: `section_id`, `doc_id`, `text`, `score`, `metadata`. `text` is the section text, including its heading line. `metadata` is what the vector store returned for the match. With the built-in stores this is `{"doc_id": ...}`.
- `Document`: `tenant_id`, `doc_id`, `title`, `metadata`, `created_at`.
- `Section`: `tenant_id`, `section_id`, `doc_id`, `text`, `metadata`, `created_at`.
- `SplitSection`: output of a splitter. `text`, `heading_level`, `section_number`, `section_title`, `parent_path`, `chunk_index`, `total_chunks`.

### Tenant context

- `tenant_context(tenant_id)`: context manager. Use with `with` or `async with`, or as a decorator on a sync or async function. The tenant applies only while the decorated code runs. Decorating a generator raises `TypeError`: enter `tenant_context` in the code that iterates it. An object cannot be entered again while it is active, so blocks that may overlap (nested blocks, tasks or threads) each need their own `tenant_context(...)`; a decorated function gets a fresh one per call.
- `get_current_tenant()`: the current tenant ID, or `None`.
- `require_tenant()`: the current tenant ID. Raises `RuntimeError` if none is set.

---

## Vector Stores

### In-memory (default)

`InMemoryVectorStore` keeps vectors in a Python dict, separated by tenant, and ranks by cosine similarity. The first upsert sets the vector length for the store. An upsert or query vector of a different length raises `ValueError`, and a rejected upsert stores nothing. Vectors are lost when the process exits, while documents and sections remain in SQLite. Use it for tests and development.

### Qdrant

Requires the `qdrant` extra and a running Qdrant server. Not exported from the top-level package. `embed` here and below is the function from Quick Start:

```python
from raag import Raag
from raag.vector.qdrant import QdrantVectorStore

store = QdrantVectorStore(
    url="http://localhost:6333",   # default
    collection_name="raag",        # default; created if missing
    vector_size=5,                 # default 1024; must match the length of embed's vectors
)
raag = Raag(embed_fn=embed, db_path="raag.db", vector_store=store)
```

`vector_size` must equal the length of the vectors your `embed_fn` returns. The Quick Start `embed` returns 5 values per text; a real embedding model usually returns hundreds or more. If the sizes differ, `add_document` fails with an error from Qdrant and keeps nothing from that document. The size is applied only when the collection is created, so an existing collection keeps its original size.

All tenants share one collection. Searches and deletes filter on a `tenant_id` payload field, so one tenant cannot delete another tenant's points. Store operations use qdrant-client's `AsyncQdrantClient` and do not block the event loop; the constructor checks or creates the collection synchronously. The collection uses cosine distance.

### Bring your own

`VectorStore` is a protocol with three async methods. Any class that implements them works, with no base class needed:

```python
from raag import Raag, VectorItem, VectorSearchResult


class MyVectorStore:
    async def upsert(self, items: list[VectorItem]) -> None: ...
    async def search(self, vector: list[float], k: int, tenant_id: str) -> list[VectorSearchResult]: ...
    async def delete(self, ids: list[str], tenant_id: str) -> None: ...


raag = Raag(embed_fn=embed, vector_store=MyVectorStore())
```

`VectorItem` has `id`, `vector`, `metadata` and `tenant_id`. `search` must return only the given tenant's vectors, sorted by score, highest first. `add_document` does not call `upsert` for a document with no sections, and calls `delete` with the document's section IDs if storing it fails.

---

## What was planned

These were planned and were not built. They are listed for context only:

- Relationship extraction between sections and documents (references, overrides, supersedes, defined terms and similar), derived from structure and explicit patterns rather than an LLM.
- A document relationship graph and graph-guided retrieval that expands search results along those relationships.
- Document versioning and conflict detection.
- Parsing of formats beyond plain text and Markdown.

---

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,qdrant]"
pytest
```

---

## License

MIT. See [LICENSE](LICENSE).

*RAAG: Relationship-Aware Augmented Generation. By [AALA AI](https://aala.ai).*
