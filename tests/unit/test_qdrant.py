"""Unit tests for QdrantVectorStore using qdrant-client local mode (no server)."""

import uuid

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from raag import Raag, tenant_context
from raag.protocols import VectorItem
from raag.vector import qdrant as qdrant_module
from raag.vector.qdrant import QdrantVectorStore


class _CountingClient:
    """Wraps a sync client and counts data calls made through it."""

    def __init__(self, client):
        self._client = client
        self.data_calls = 0

    def __getattr__(self, name):
        if name in ("upsert", "query_points", "delete", "search"):
            self.data_calls += 1
        return getattr(self._client, name)


@pytest.fixture
async def local_store(tmp_path, monkeypatch):
    """A QdrantVectorStore backed by on-disk local mode instead of a server."""
    path = str(tmp_path / "qdrant")
    sync_clients = []

    def sync_factory(url):
        client = _CountingClient(qdrant_client.QdrantClient(path=path))
        sync_clients.append(client)
        return client

    monkeypatch.setattr(qdrant_module, "QdrantClient", sync_factory)
    monkeypatch.setattr(
        qdrant_module,
        "AsyncQdrantClient",
        lambda url: qdrant_client.AsyncQdrantClient(path=path),
        raising=False,
    )
    store = QdrantVectorStore(url="http://unused:6333", collection_name="test", vector_size=3)
    yield store, sync_clients
    await store._client.close()


def _item(tenant_id, vector):
    return VectorItem(id=str(uuid.uuid4()), vector=vector, metadata={"doc_id": "d"}, tenant_id=tenant_id)


async def test_delete_cannot_remove_other_tenants_points(local_store):
    store, _ = local_store
    a1 = _item("tenant_a", [1.0, 0.0, 0.0])
    a2 = _item("tenant_a", [0.0, 1.0, 0.0])
    b1 = _item("tenant_b", [1.0, 0.0, 0.0])
    await store.upsert([a1, a2, b1])

    # tenant_b names tenant_a's IDs: nothing is deleted
    await store.delete([a1.id, a2.id], "tenant_b")
    assert {r.id for r in await store.search([1.0, 0.0, 0.0], 10, "tenant_a")} == {a1.id, a2.id}

    # A mixed list deletes only the caller's own points
    await store.delete([a1.id, b1.id], "tenant_a")
    assert [r.id for r in await store.search([1.0, 0.0, 0.0], 10, "tenant_a")] == [a2.id]
    assert [r.id for r in await store.search([1.0, 0.0, 0.0], 10, "tenant_b")] == [b1.id]


async def test_operations_use_async_client(local_store):
    store, sync_clients = local_store
    assert isinstance(store._client, qdrant_client.AsyncQdrantClient)

    item = _item("tenant_a", [1.0, 0.0, 0.0])
    await store.upsert([item])
    results = await store.search([1.0, 0.0, 0.0], 10, "tenant_a")
    await store.delete([item.id], "tenant_a")

    assert [r.id for r in results] == [item.id]
    assert results[0].metadata == {"doc_id": "d"}
    assert await store.search([1.0, 0.0, 0.0], 10, "tenant_a") == []
    assert sum(c.data_calls for c in sync_clients) == 0


async def test_empty_upsert_is_noop(local_store):
    store, _ = local_store
    await store.upsert([])
    assert await store.search([1.0, 0.0, 0.0], 10, "tenant_a") == []


async def _embed_4d(texts):
    return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


async def _embed_3d(texts):
    return [[1.0, 0.0, 0.0] for _ in texts]


async def test_add_document_with_wrong_vector_size_leaves_nothing(local_store, tmp_path):
    store, _ = local_store
    raag = Raag(embed_fn=_embed_4d, db_path=str(tmp_path / "q.db"), vector_store=store)

    async with tenant_context("tenant_a"):
        with pytest.raises(Exception):
            await raag.add_document("# A\n\nOne.\n\n# B\n\nTwo.", {"title": "Wrong size"})
        assert await raag.list_documents() == []
    assert await store.search([1.0, 0.0, 0.0], 10, "tenant_a") == []


async def test_add_empty_document(local_store, tmp_path):
    store, _ = local_store
    raag = Raag(embed_fn=_embed_3d, db_path=str(tmp_path / "q.db"), vector_store=store)

    async with tenant_context("tenant_a"):
        doc_id = await raag.add_document("", {"title": "Empty"})
        assert [d.doc_id for d in await raag.list_documents()] == [doc_id]
        assert await raag.search([1.0, 0.0, 0.0]) == []
