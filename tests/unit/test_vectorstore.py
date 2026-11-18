"""Unit tests for InMemoryVectorStore."""

import pytest
import math

from raag.protocols import VectorItem, VectorSearchResult
from raag.vector.memory import InMemoryVectorStore


@pytest.mark.asyncio
async def test_upsert_stores_vectors_correctly():
    """Test that upsert stores vectors in the correct tenant."""
    store = InMemoryVectorStore()

    items = [
        VectorItem(
            id="doc1",
            vector=[1.0, 0.0, 0.0],
            metadata={"title": "Doc 1"},
            tenant_id="tenant_a"
        ),
        VectorItem(
            id="doc2",
            vector=[0.0, 1.0, 0.0],
            metadata={"title": "Doc 2"},
            tenant_id="tenant_a"
        ),
    ]

    await store.upsert(items)

    # Verify vectors are stored
    results = await store.search([1.0, 0.0, 0.0], k=10, tenant_id="tenant_a")

    assert len(results) == 2
    # First result should be doc1 with perfect similarity
    assert results[0].id == "doc1"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata == {"title": "Doc 1"}


@pytest.mark.asyncio
async def test_upsert_updates_existing_vector():
    """Test that upsert updates an existing vector."""
    store = InMemoryVectorStore()

    # Insert initial vector
    await store.upsert([
        VectorItem(
            id="doc1",
            vector=[1.0, 0.0, 0.0],
            metadata={"version": 1},
            tenant_id="tenant_a"
        )
    ])

    # Update with new vector and metadata
    await store.upsert([
        VectorItem(
            id="doc1",
            vector=[0.0, 1.0, 0.0],
            metadata={"version": 2},
            tenant_id="tenant_a"
        )
    ])

    results = await store.search([0.0, 1.0, 0.0], k=10, tenant_id="tenant_a")

    assert len(results) == 1
    assert results[0].id == "doc1"
    assert results[0].score == pytest.approx(1.0)
    assert results[0].metadata == {"version": 2}


@pytest.mark.asyncio
async def test_search_returns_correct_results_with_cosine_similarity():
    """Test that search uses cosine similarity correctly."""
    store = InMemoryVectorStore()

    # Orthogonal vectors: [1,0,0] and [0,1,0] have similarity 0
    # Identical vectors have similarity 1
    await store.upsert([
        VectorItem(id="orthogonal", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="identical", vector=[0.0, 1.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="partial", vector=[0.0, 1.0, 1.0], metadata={}, tenant_id="tenant_a"),
    ])

    results = await store.search([0.0, 1.0, 0.0], k=10, tenant_id="tenant_a")

    assert len(results) == 3
    # identical vector should be first (similarity = 1)
    assert results[0].id == "identical"
    assert results[0].score == pytest.approx(1.0)
    # partial vector should be second (similarity = 1/sqrt(2) ≈ 0.707)
    assert results[1].id == "partial"
    assert results[1].score == pytest.approx(1.0 / math.sqrt(2))
    # orthogonal vector should be last (similarity = 0)
    assert results[2].id == "orthogonal"
    assert results[2].score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_respects_k_parameter():
    """Test that search returns only k results."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id=f"doc{i}", vector=[float(i), 0.0, 0.0], metadata={}, tenant_id="tenant_a")
        for i in range(10)
    ])

    results = await store.search([1.0, 0.0, 0.0], k=3, tenant_id="tenant_a")

    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_returns_empty_list_for_nonexistent_tenant():
    """Test that search returns empty list when tenant doesn't exist."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="doc1", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a")
    ])

    results = await store.search([1.0, 0.0], k=10, tenant_id="tenant_b")

    assert results == []


@pytest.mark.asyncio
async def test_tenant_isolation():
    """Test that tenant A cannot search tenant B's vectors."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="a_doc1", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="b_doc1", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_b"),
    ])

    # Search tenant_a - should only see tenant_a's docs
    results_a = await store.search([1.0, 0.0], k=10, tenant_id="tenant_a")
    assert len(results_a) == 1
    assert results_a[0].id == "a_doc1"

    # Search tenant_b - should only see tenant_b's docs
    results_b = await store.search([1.0, 0.0], k=10, tenant_id="tenant_b")
    assert len(results_b) == 1
    assert results_b[0].id == "b_doc1"


@pytest.mark.asyncio
async def test_delete_removes_vectors():
    """Test that delete removes vectors by ID."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="doc1", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="doc2", vector=[0.0, 1.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="doc3", vector=[0.0, 0.0, 1.0], metadata={}, tenant_id="tenant_a"),
    ])

    # Delete doc2
    await store.delete(["doc2"], tenant_id="tenant_a")

    results = await store.search([1.0, 1.0, 1.0], k=10, tenant_id="tenant_a")

    assert len(results) == 2
    result_ids = {r.id for r in results}
    assert "doc2" not in result_ids
    assert "doc1" in result_ids
    assert "doc3" in result_ids


@pytest.mark.asyncio
async def test_delete_nonexistent_id_is_noop():
    """Test that deleting a non-existent ID doesn't raise an error."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="doc1", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a")
    ])

    # Delete non-existent ID
    await store.delete(["nonexistent"], tenant_id="tenant_a")

    # doc1 should still be there
    results = await store.search([1.0, 0.0], k=10, tenant_id="tenant_a")
    assert len(results) == 1
    assert results[0].id == "doc1"


@pytest.mark.asyncio
async def test_delete_from_nonexistent_tenant_is_noop():
    """Test that deleting from a non-existent tenant doesn't raise an error."""
    store = InMemoryVectorStore()

    # Should not raise
    await store.delete(["doc1"], tenant_id="nonexistent_tenant")


@pytest.mark.asyncio
async def test_delete_multiple_ids():
    """Test that delete can remove multiple IDs at once."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="doc1", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="doc2", vector=[0.0, 1.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="doc3", vector=[0.0, 0.0, 1.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="doc4", vector=[1.0, 1.0, 0.0], metadata={}, tenant_id="tenant_a"),
    ])

    # Delete doc2 and doc4
    await store.delete(["doc2", "doc4"], tenant_id="tenant_a")

    results = await store.search([1.0, 1.0, 1.0], k=10, tenant_id="tenant_a")

    assert len(results) == 2
    result_ids = {r.id for r in results}
    assert result_ids == {"doc1", "doc3"}


@pytest.mark.asyncio
async def test_cosine_similarity_zero_vectors():
    """Test cosine similarity with zero vectors."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="zero", vector=[0.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="one", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a"),
    ])

    results = await store.search([1.0, 0.0], k=10, tenant_id="tenant_a")

    assert len(results) == 2
    # zero vector should have similarity 0
    zero_result = next(r for r in results if r.id == "zero")
    assert zero_result.score == 0.0
    # one vector should have similarity 1
    one_result = next(r for r in results if r.id == "one")
    assert one_result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_multiple_tenants_independent():
    """Test that multiple tenants can have the same doc IDs without conflict."""
    store = InMemoryVectorStore()

    # Same doc ID in different tenants
    await store.upsert([
        VectorItem(id="shared_id", vector=[1.0, 0.0], metadata={"tenant": "a"}, tenant_id="tenant_a"),
        VectorItem(id="shared_id", vector=[0.0, 1.0], metadata={"tenant": "b"}, tenant_id="tenant_b"),
    ])

    # Each tenant should see their own version
    results_a = await store.search([1.0, 0.0], k=10, tenant_id="tenant_a")
    assert len(results_a) == 1
    assert results_a[0].metadata == {"tenant": "a"}

    results_b = await store.search([0.0, 1.0], k=10, tenant_id="tenant_b")
    assert len(results_b) == 1
    assert results_b[0].metadata == {"tenant": "b"}


@pytest.mark.asyncio
async def test_search_results_sorted_by_score_descending():
    """Test that search results are sorted by score (highest first)."""
    store = InMemoryVectorStore()

    # Use 2D vectors at different angles to get different cosine similarities
    # Query is [1, 0], so:
    # - "high": [1, 0] = identical -> similarity = 1.0
    # - "medium": [1, 1] = 45 degrees -> similarity = 1/sqrt(2) ≈ 0.707
    # - "low": [0, 1] = orthogonal -> similarity = 0.0
    await store.upsert([
        VectorItem(id="low", vector=[0.0, 1.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="high", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a"),
        VectorItem(id="medium", vector=[1.0, 1.0], metadata={}, tenant_id="tenant_a"),
    ])

    results = await store.search([1.0, 0.0], k=10, tenant_id="tenant_a")

    assert len(results) == 3
    assert results[0].id == "high"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].id == "medium"
    assert results[1].score == pytest.approx(1.0 / math.sqrt(2))
    assert results[2].id == "low"
    assert results[2].score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_search_dimension_mismatch_raises():
    """Test that a query and stored vector of different lengths fail loudly."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="doc1", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a"),
    ])

    with pytest.raises(ValueError, match="dimension mismatch"):
        await store.search([1.0, 0.0], k=10, tenant_id="tenant_a")


@pytest.mark.asyncio
async def test_upsert_dimension_mismatch_raises_and_stores_nothing():
    """Test that a vector of the wrong length is rejected before anything is stored."""
    store = InMemoryVectorStore()

    await store.upsert([
        VectorItem(id="doc1", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a"),
    ])

    with pytest.raises(ValueError, match="dimension mismatch"):
        await store.upsert([
            VectorItem(id="doc2", vector=[0.0, 1.0, 0.0], metadata={}, tenant_id="tenant_a"),
            VectorItem(id="bad", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a"),
        ])
    with pytest.raises(ValueError, match="dimension mismatch"):
        await store.upsert([
            VectorItem(id="other", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_b"),
        ])

    results = await store.search([1.0, 0.0, 0.0], k=10, tenant_id="tenant_a")
    assert [r.id for r in results] == ["doc1"]
    assert await store.search([1.0, 0.0, 0.0], k=10, tenant_id="tenant_b") == []


@pytest.mark.asyncio
async def test_upsert_mixed_lengths_in_first_batch_raises():
    """Test that the first batch must agree on one dimension."""
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="dimension mismatch"):
        await store.upsert([
            VectorItem(id="a", vector=[1.0, 0.0], metadata={}, tenant_id="tenant_a"),
            VectorItem(id="b", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a"),
        ])
    assert await store.search([1.0, 0.0], k=10, tenant_id="tenant_a") == []
    await store.upsert([VectorItem(id="c", vector=[1.0, 0.0, 0.0], metadata={}, tenant_id="tenant_a")])
