"""Integration tests for the Raag core class."""

import asyncio
import concurrent.futures
import threading
import time

import pytest
from pytest import fixture
from sqlalchemy import select

from raag import Raag, Document, InMemoryVectorStore, tenant_context
from raag.models.results import SearchResult
from raag.storage.models import DocumentModel, SectionModel


KEYWORDS = ["fox", "cat", "bird"]


async def _keyword_embed(texts: list[str]) -> list[list[float]]:
    """Embed by keyword presence so each topic gets a distinct direction."""
    return [
        [1.0 if word in t.lower() else 0.0 for word in KEYWORDS] + [0.01]
        for t in texts
    ]


def _query(word: str) -> list[float]:
    return [1.0 if w == word else 0.0 for w in KEYWORDS] + [0.0]


@fixture
async def mock_embed_fn():
    """Mock embedding function for testing."""
    async def embed(texts: list[str]) -> list[list[float]]:
        # Return simple embeddings based on text length
        # For consistent results, use a deterministic approach
        return [[len(t) * 0.1] * 10 for t in texts]
    return embed


@fixture
async def raag_instance(tmp_path, mock_embed_fn):
    """Create a Raag instance for testing."""
    db_path = str(tmp_path / "test.db")
    return Raag(embed_fn=mock_embed_fn, db_path=db_path)


@fixture
def keyword_raag(tmp_path):
    """Create a Raag instance with keyword embeddings."""
    return Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "keyword.db"))


async def _db_rows(raag: Raag, doc_id: str | None = None):
    """Return document and section rows for a doc_id, or all rows, across all tenants."""
    async def load(session):
        doc_query, section_query = select(DocumentModel), select(SectionModel)
        if doc_id is not None:
            doc_query = doc_query.where(DocumentModel.doc_id == doc_id)
            section_query = section_query.where(SectionModel.doc_id == doc_id)
        docs = (await session.execute(doc_query)).scalars().all()
        sections = (await session.execute(section_query)).scalars().all()
        return list(docs), list(sections)
    return await raag._db(load)


ANIMALS = "# Fox\n\nThe fox runs.\n\n# Cat\n\nThe cat sleeps.\n\n# Bird\n\nThe bird sings."


async def test_add_document_creates_document_and_sections(raag_instance):
    """Test that add_document creates document and sections in DB."""
    async with tenant_context("tenant_1"):
        doc_id = await raag_instance.add_document(
            "# One\n\nSection 1\n\n# Two\n\nSection 2\n\n# Three\n\nSection 3",
            {"title": "Test Doc", "source_path": "docs/test.md"}
        )

    assert isinstance(doc_id, str)
    docs, sections = await _db_rows(raag_instance, doc_id)

    assert len(docs) == 1
    assert docs[0].tenant_id == "tenant_1"
    assert docs[0].title == "Test Doc"
    assert docs[0].meta == {"title": "Test Doc", "source_path": "docs/test.md"}

    assert len(sections) == 3
    assert all(s.tenant_id == "tenant_1" for s in sections)
    texts = " ".join(s.text for s in sections)
    for part in ("Section 1", "Section 2", "Section 3"):
        assert part in texts
    assert all(s.meta["source_path"] == "docs/test.md" for s in sections)


async def test_search_returns_relevant_results(keyword_raag):
    """Test that search ranks the matching section first."""
    async with tenant_context("tenant_1"):
        doc_id = await keyword_raag.add_document(ANIMALS, {"title": "Animals"})

        for word in KEYWORDS:
            results = await keyword_raag.search(_query(word), k=3)

            assert len(results) == 3
            assert all(isinstance(r, SearchResult) for r in results)
            assert all(r.doc_id == doc_id for r in results)
            assert word in results[0].text.lower()
            assert all(word not in r.text.lower() for r in results[1:])
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)
            assert scores[0] > scores[1]

        top_one = await keyword_raag.search(_query("cat"), k=1)
        assert len(top_one) == 1
        assert "cat" in top_one[0].text.lower()


async def test_delete_document_removes_all_data(keyword_raag):
    """Test that delete_document removes DB rows and vectors."""
    async with tenant_context("tenant_1"):
        doc_id = await keyword_raag.add_document(ANIMALS, {"title": "Delete Me"})
        keep_id = await keyword_raag.add_document("# Fox\n\nAnother fox.", {"title": "Keep"})

        docs, sections = await _db_rows(keyword_raag, doc_id)
        assert len(docs) == 1
        assert len(sections) == 3
        section_ids = {s.id for s in await keyword_raag.vector_store.search(_query("fox"), 100, "tenant_1")}
        assert {s.section_id for s in sections} <= section_ids

        await keyword_raag.delete_document(doc_id)

        docs, sections = await _db_rows(keyword_raag, doc_id)
        assert docs == []
        assert sections == []

        remaining = await keyword_raag.vector_store.search(_query("fox"), 100, "tenant_1")
        _, keep_sections = await _db_rows(keyword_raag, keep_id)
        assert {r.id for r in remaining} == {s.section_id for s in keep_sections}

        results = await keyword_raag.search(_query("cat"), k=10)
        assert all(r.doc_id == keep_id for r in results)
        assert [d.doc_id for d in await keyword_raag.list_documents()] == [keep_id]


async def test_tenant_isolation_in_all_operations(keyword_raag):
    """Test that tenant isolation holds for add, search, list and delete."""
    async with tenant_context("tenant_1"):
        doc_id_1 = await keyword_raag.add_document("Tenant 1 fox data", {"title": "Tenant 1 Doc"})

    async with tenant_context("tenant_2"):
        # Tenant 2 sees nothing of tenant 1
        assert await keyword_raag.search(_query("fox"), k=10) == []
        assert await keyword_raag.list_documents() == []

        doc_id_2 = await keyword_raag.add_document("Tenant 2 fox data", {"title": "Tenant 2 Doc"})

        results = await keyword_raag.search(_query("fox"), k=10)
        assert [r.doc_id for r in results] == [doc_id_2]
        assert [d.doc_id for d in await keyword_raag.list_documents()] == [doc_id_2]

        # Tenant 2 cannot delete tenant 1's document
        await keyword_raag.delete_document(doc_id_1)

    docs, sections = await _db_rows(keyword_raag, doc_id_1)
    assert len(docs) == 1
    assert len(sections) == 1
    assert len(await keyword_raag.vector_store.search(_query("fox"), 10, "tenant_1")) == 1

    async with tenant_context("tenant_1"):
        results = await keyword_raag.search(_query("fox"), k=10)
        assert [r.doc_id for r in results] == [doc_id_1]
        assert [d.doc_id for d in await keyword_raag.list_documents()] == [doc_id_1]

        # Tenant 1 cannot delete tenant 2's document
        await keyword_raag.delete_document(doc_id_2)

    docs, sections = await _db_rows(keyword_raag, doc_id_2)
    assert len(docs) == 1
    assert len(sections) == 1

    async with tenant_context("tenant_2"):
        results = await keyword_raag.search(_query("fox"), k=10)
        assert [r.doc_id for r in results] == [doc_id_2]


async def test_list_documents_returns_current_tenant_documents(keyword_raag):
    """Test list_documents returns only the current tenant's documents."""
    async with tenant_context("tenant_1"):
        assert await keyword_raag.list_documents() == []
        first = await keyword_raag.add_document("Fox", {"title": "First", "tag": "a"})
        second = await keyword_raag.add_document("Cat", {"title": "Second"})

    async with tenant_context("tenant_2"):
        await keyword_raag.add_document("Bird", {"title": "Other"})

    async with tenant_context("tenant_1"):
        docs = await keyword_raag.list_documents()

    assert all(isinstance(d, Document) for d in docs)
    assert {d.doc_id for d in docs} == {first, second}
    assert all(d.tenant_id == "tenant_1" for d in docs)
    by_id = {d.doc_id: d for d in docs}
    assert by_id[first].title == "First"
    assert by_id[first].metadata == {"title": "First", "tag": "a"}


async def test_list_documents_requires_tenant(keyword_raag):
    """Test list_documents fails without tenant context."""
    with pytest.raises(RuntimeError, match="No tenant context set"):
        await keyword_raag.list_documents()


def test_add_document_sync_wrapper(keyword_raag):
    """Test add_document_sync from plain sync code."""
    with tenant_context("tenant_1"):
        doc_id = keyword_raag.add_document_sync(ANIMALS, {"title": "Sync Doc"})
        results = keyword_raag.search_sync(_query("bird"), k=3)

    assert isinstance(doc_id, str)
    assert len(results) == 3
    assert results[0].doc_id == doc_id
    assert "bird" in results[0].text.lower()


def test_search_sync_wrapper(keyword_raag):
    """Test search_sync from plain sync code respects tenant context."""
    with tenant_context("tenant_1"):
        doc_id = keyword_raag.add_document_sync(ANIMALS, {"title": "Sync Doc"})
        results = keyword_raag.search_sync(_query("cat"))

    assert len(results) == 3
    assert "cat" in results[0].text.lower()
    assert all(r.doc_id == doc_id for r in results)

    with tenant_context("tenant_2"):
        assert keyword_raag.search_sync(_query("cat")) == []


def test_delete_document_sync_wrapper(keyword_raag):
    """Test delete_document_sync from plain sync code."""
    with tenant_context("tenant_1"):
        doc_id = keyword_raag.add_document_sync(ANIMALS, {"title": "Sync Doc"})

    with tenant_context("tenant_2"):
        keyword_raag.delete_document_sync(doc_id)
    with tenant_context("tenant_1"):
        assert len(keyword_raag.search_sync(_query("fox"))) == 3
        keyword_raag.delete_document_sync(doc_id)
        assert keyword_raag.search_sync(_query("fox")) == []


def test_sync_wrappers_require_tenant(keyword_raag):
    """Test sync wrappers fail without tenant context."""
    with pytest.raises(RuntimeError, match="No tenant context set"):
        keyword_raag.add_document_sync("Fox", {"title": "No Tenant"})
    with pytest.raises(RuntimeError, match="No tenant context set"):
        keyword_raag.search_sync(_query("fox"))


async def test_sync_wrappers_inside_running_loop(keyword_raag):
    """Test sync wrappers keep the tenant when called inside an event loop."""
    async with tenant_context("tenant_1"):
        doc_id = keyword_raag.add_document_sync(ANIMALS, {"title": "Loop Doc"})
        results = keyword_raag.search_sync(_query("fox"), k=3)
        assert results[0].doc_id == doc_id
        assert "fox" in results[0].text.lower()

    with tenant_context("tenant_2"):
        assert keyword_raag.search_sync(_query("fox")) == []
        keyword_raag.delete_document_sync(doc_id)

    async with tenant_context("tenant_1"):
        assert len(await keyword_raag.search(_query("fox"), k=3)) == 3
        keyword_raag.delete_document_sync(doc_id)
        assert await keyword_raag.search(_query("fox"), k=3) == []
        docs, sections = await _db_rows(keyword_raag, doc_id)
        assert docs == []
        assert sections == []


async def test_empty_search_returns_empty_list(raag_instance):
    """Test that search with no data returns empty list."""
    async with tenant_context("tenant_1"):
        results = await raag_instance.search([0.5] * 10, k=10)
        assert results == []


async def test_full_workflow(raag_instance):
    """Test the complete workflow: add, search, delete."""
    async with tenant_context("tenant_1"):
        # Add
        doc_id = await raag_instance.add_document(
            "First paragraph\n\nSecond paragraph\n\nThird paragraph",
            {"title": "Workflow Test"}
        )
        assert doc_id is not None

        # Search
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) >= 1

        # Delete
        await raag_instance.delete_document(doc_id)

        # Verify deletion
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 0


async def test_add_document_with_empty_metadata(raag_instance):
    """Test add_document with minimal metadata."""
    async with tenant_context("tenant_1"):
        doc_id = await raag_instance.add_document("Test content", {})
        assert doc_id is not None


async def test_add_document_splits_into_sections(raag_instance):
    """Test that markdown text is split into sections by heading."""
    async with tenant_context("tenant_1"):
        # Text with 3 headed sections (MarkdownSplitter splits by heading)
        text = "# Section one\n\nBody one.\n\n# Section two\n\nBody two.\n\n# Section three\n\nBody three."
        doc_id = await raag_instance.add_document(text, {"title": "Split Test"})

        # Search should return results from all 3 sections
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 3


async def test_custom_vector_store_and_config(raag_instance, mock_embed_fn):
    """Test Raag initialization with custom config."""
    from raag import RaagConfig, InMemoryVectorStore

    config = RaagConfig(embed_batch_size=50, default_k=5)
    vector_store = InMemoryVectorStore()

    raag = Raag(
        embed_fn=mock_embed_fn,
        db_path=":memory:",
        vector_store=vector_store,
        config=config
    )

    async with tenant_context("tenant_1"):
        doc_id = await raag.add_document("Test", {"title": "Custom Config"})
        assert doc_id is not None

        results = await raag.search([0.5] * 10, k=5)
        assert len(results) > 0


async def test_concurrent_first_calls_create_tables_once(tmp_path):
    """Test that concurrent first calls on a fresh instance do not race on table creation."""
    raag = Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "race.db"))

    async def add(tenant_id):
        async with tenant_context(tenant_id):
            return await raag.add_document("# Fox\n\nThe fox runs.", {"title": tenant_id})

    doc_ids = await asyncio.gather(*(add(f"tenant_{i}") for i in range(6)))
    assert len(set(doc_ids)) == 6

    async with tenant_context("tenant_3"):
        assert [d.doc_id for d in await raag.list_documents()] == [doc_ids[3]]


def test_concurrent_first_sync_calls_from_threads(tmp_path):
    """Test that sync wrappers in several threads, each with its own loop, share one fresh database."""
    db_path = str(tmp_path / "threads.db")
    instances = [Raag(embed_fn=_keyword_embed, db_path=db_path) for _ in range(6)]
    barrier = threading.Barrier(len(instances))

    def add(i):
        barrier.wait()
        with tenant_context(f"tenant_{i}"):
            return instances[i].add_document_sync("# Fox\n\nThe fox runs.", {"title": str(i)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(instances)) as pool:
        doc_ids = list(pool.map(add, range(len(instances))))
    assert len(set(doc_ids)) == len(instances)


async def test_add_document_embeds_in_batches(tmp_path):
    """Test that add_document sends at most embed_batch_size texts per embed_fn call."""
    from raag import RaagConfig

    calls = []

    async def embed(texts):
        calls.append(list(texts))
        return await _keyword_embed(texts)

    raag = Raag(embed_fn=embed, db_path=str(tmp_path / "batch.db"), config=RaagConfig(embed_batch_size=2))
    text = "\n\n".join(f"# Part {i}\n\nThe fox {i}." for i in range(5))

    async with tenant_context("tenant_1"):
        doc_id = await raag.add_document(text, {"title": "Batched"})
        results = await raag.search(_query("fox"), k=10)

    assert [len(c) for c in calls] == [2, 2, 1]
    assert [t.split("\n")[0] for c in calls for t in c] == [f"# Part {i}" for i in range(5)]
    assert len(results) == 5
    assert all(r.doc_id == doc_id for r in results)


async def test_add_document_rejects_invalid_batch_size(tmp_path):
    """Test that a batch size below 1 fails before anything is stored."""
    from raag import RaagConfig

    raag = Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "bad.db"), config=RaagConfig(embed_batch_size=0))
    async with tenant_context("tenant_1"):
        with pytest.raises(ValueError, match="embed_batch_size"):
            await raag.add_document("# Fox\n\nThe fox runs.", {"title": "Bad"})
        assert await raag.list_documents() == []


class _FailingUpsertStore(InMemoryVectorStore):
    """Stores the vectors, then fails, like a backend erroring mid-request."""

    def __init__(self):
        super().__init__()
        self.deleted = []

    async def upsert(self, items):
        await super().upsert(items)
        raise RuntimeError("upsert failed")

    async def delete(self, ids, tenant_id):
        self.deleted.append((list(ids), tenant_id))
        await super().delete(ids, tenant_id)


class _NoEmptyUpsertStore(InMemoryVectorStore):
    """Rejects an empty upsert, as Qdrant does."""

    async def upsert(self, items):
        if not items:
            raise ValueError("empty upsert")
        await super().upsert(items)


async def test_failed_vector_upsert_leaves_nothing_behind(tmp_path):
    """Test that add_document removes its rows and vectors when the vector store fails."""
    store = _FailingUpsertStore()
    raag = Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "fail.db"), vector_store=store)

    async with tenant_context("tenant_1"):
        with pytest.raises(RuntimeError, match="upsert failed"):
            await raag.add_document(ANIMALS, {"title": "Fails"})
        assert await raag.list_documents() == []
        assert await store.search(_query("fox"), 10, "tenant_1") == []

    assert await _db_rows(raag) == ([], [])
    assert len(store.deleted) == 1
    assert len(store.deleted[0][0]) == 3
    assert store.deleted[0][1] == "tenant_1"


async def test_failed_section_save_removes_document(tmp_path, monkeypatch):
    """Test that a failure after the document row is written removes that row."""
    import raag.core as core

    async def failing_save_sections(session, sections):
        raise RuntimeError("sections failed")

    monkeypatch.setattr(core, "save_sections", failing_save_sections)
    raag = Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "fail.db"))

    async with tenant_context("tenant_1"):
        with pytest.raises(RuntimeError, match="sections failed"):
            await raag.add_document(ANIMALS, {"title": "Fails"})
    assert await _db_rows(raag) == ([], [])


async def test_cleanup_finishes_when_cancelled_again(tmp_path, monkeypatch):
    """Test that a second cancel during cleanup does not leave rows or vectors behind."""
    import raag.core as core

    upserted = asyncio.Event()
    cleanup_started, release = threading.Event(), threading.Event()
    real_delete = core.db_delete_document

    class HangingStore(InMemoryVectorStore):
        async def upsert(self, items):
            await super().upsert(items)
            upserted.set()
            await asyncio.Event().wait()

    async def slow_delete(session, doc_id):
        cleanup_started.set()
        await asyncio.to_thread(release.wait, 30)
        return await real_delete(session, doc_id)

    monkeypatch.setattr(core, "db_delete_document", slow_delete)
    store = HangingStore()
    raag = Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "cancel.db"), vector_store=store)

    async with tenant_context("tenant_1"):
        task = asyncio.create_task(raag.add_document(ANIMALS, {"title": "Cancelled"}))
        await asyncio.wait_for(upserted.wait(), 30)
        assert len(await store.search(_query("fox"), 10, "tenant_1")) == 3
        task.cancel()
        assert await asyncio.to_thread(cleanup_started.wait, 30)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)

        assert await store.search(_query("fox"), 10, "tenant_1") == []
    assert await _db_rows(raag) == ([], [])


@pytest.mark.parametrize("store_cls", [InMemoryVectorStore, _NoEmptyUpsertStore])
@pytest.mark.parametrize("text", ["", "  \n\n  "])
async def test_empty_document_is_stored_without_vectors(tmp_path, store_cls, text):
    """Test that an empty document behaves the same with every vector store."""
    raag = Raag(embed_fn=_keyword_embed, db_path=str(tmp_path / "empty.db"), vector_store=store_cls())

    async with tenant_context("tenant_1"):
        doc_id = await raag.add_document(text, {"title": "Empty"})
        assert [d.doc_id for d in await raag.list_documents()] == [doc_id]
        assert await raag.search(_query("fox")) == []
        await raag.delete_document(doc_id)
        assert await raag.list_documents() == []


@pytest.mark.parametrize("returned", [2, 4])
async def test_wrong_embedding_count_is_rejected(tmp_path, returned):
    """Test that embed_fn returning the wrong number of vectors fails before anything is stored."""
    async def embed(texts):
        vectors = await _keyword_embed(texts)
        return (vectors * 2)[:returned]

    store = InMemoryVectorStore()
    raag = Raag(embed_fn=embed, db_path=str(tmp_path / "count.db"), vector_store=store)

    async with tenant_context("tenant_1"):
        with pytest.raises(ValueError, match=f"embed_fn returned {returned} vectors for 3 texts"):
            await raag.add_document(ANIMALS, {"title": "Bad"})
        assert await raag.list_documents() == []
        assert await store.search(_query("fox"), 10, "tenant_1") == []
    assert await _db_rows(raag) == ([], [])


def _run_threads(target, count, timeout=60):
    """Run target(i) in daemon threads so a hang fails the test instead of blocking it."""
    results, errors = [None] * count, []

    def run(i):
        try:
            results[i] = target(i)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(count)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "threads hung"
    if errors:
        raise errors[0]
    return results


@pytest.mark.parametrize("db_name", ["file", ":memory:"])
def test_one_instance_shared_by_threads(tmp_path, db_name):
    """Test that threads sharing one fresh instance do not hang and keep their tenants apart."""
    workers, rounds = 8, 5
    db_path = db_name if db_name == ":memory:" else str(tmp_path / "shared.db")
    raag = Raag(embed_fn=_keyword_embed, db_path=db_path)
    owned = {}

    for round_no in range(rounds):
        barrier = threading.Barrier(workers)

        def work(i):
            tenant_id = f"tenant_{i}"
            barrier.wait(timeout=30)
            with tenant_context(tenant_id):
                doc_id = raag.add_document_sync(f"# Fox\n\nThe fox {round_no} {i}.", {"title": tenant_id})
                results = raag.search_sync(_query("fox"), k=100)
            return tenant_id, doc_id, {r.doc_id for r in results}

        for tenant_id, doc_id, seen in _run_threads(work, workers, timeout=30):
            owned.setdefault(tenant_id, set()).add(doc_id)
            assert doc_id in seen
            assert seen <= owned[tenant_id]

    for tenant_id, doc_ids in owned.items():
        assert len(doc_ids) == rounds
        with tenant_context(tenant_id):
            assert {r.doc_id for r in raag.search_sync(_query("fox"), k=100)} == doc_ids


@pytest.mark.parametrize("db_name", ["file", ":memory:"])
def test_repeated_sync_calls_leave_no_loop_errors(tmp_path, monkeypatch, db_name):
    """Test that sync wrappers leave no connections behind on closed event loops."""
    import gc
    import warnings

    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    db_path = db_name if db_name == ":memory:" else str(tmp_path / "repeat.db")
    raag = Raag(embed_fn=_keyword_embed, db_path=db_path)
    gc.collect()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tenant_context("tenant_1"):
            for i in range(20):
                raag.add_document_sync(f"# Fox\n\nThe fox {i}.", {"title": str(i)})
                assert len(raag.search_sync(_query("fox"), k=100)) == i + 1
        del raag
        gc.collect()
        time.sleep(0.3)
        gc.collect()

    assert errors == []
    assert [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)] == []


@pytest.mark.parametrize("db_name", ["file", ":memory:"])
async def test_async_and_sync_calls_share_one_database(tmp_path, db_name):
    """Test async calls, sync calls inside the loop and calls from other threads on one instance."""
    db_path = db_name if db_name == ":memory:" else str(tmp_path / "mixed.db")
    raag = Raag(embed_fn=_keyword_embed, db_path=db_path)

    async def add_async(tenant_id):
        async with tenant_context(tenant_id):
            return await raag.add_document("# Fox\n\nThe fox runs.", {"title": tenant_id})

    def add_sync(tenant_id):
        with tenant_context(tenant_id):
            return raag.add_document_sync("# Fox\n\nThe fox runs.", {"title": tenant_id})

    async_ids = await asyncio.wait_for(asyncio.gather(*(add_async(f"tenant_{i}") for i in range(4))), 30)
    thread_ids = await asyncio.to_thread(_run_threads, lambda i: add_sync(f"tenant_{i}"), 4, 30)
    async with tenant_context("tenant_0"):
        inline_id = raag.add_document_sync("# Fox\n\nThe fox runs.", {"title": "inline"})

    for i in range(4):
        async with tenant_context(f"tenant_{i}"):
            expected = {async_ids[i], thread_ids[i]} | ({inline_id} if i == 0 else set())
            assert {d.doc_id for d in await raag.list_documents()} == expected
            assert {r.doc_id for r in await raag.search(_query("fox"), k=100)} == expected
