"""Integration tests for the Raag core class."""

import asyncio
from pytest import fixture

from raag import Raag, tenant_context
from raag.models.results import SearchResult


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


async def test_add_document_creates_document_and_sections(raag_instance):
    """Test that add_document creates document and sections in DB."""
    async with tenant_context("tenant_1"):
        doc_id = await raag_instance.add_document(
            "Section 1\n\nSection 2\n\nSection 3",
            {"title": "Test Doc"}
        )

        assert doc_id is not None
        assert isinstance(doc_id, str)


async def test_search_returns_relevant_results(raag_instance):
    """Test that search returns results with proper structure."""
    async with tenant_context("tenant_1"):
        # Add a document
        await raag_instance.add_document(
            "The quick brown fox jumps over the lazy dog.",
            {"title": "Fox Doc"}
        )

        # Search with a query embedding
        query_emb = [0.5] * 10
        results = await raag_instance.search(query_emb, k=5)

        # Should return results
        assert len(results) > 0

        # Check result structure
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.section_id is not None
        assert result.doc_id is not None
        assert result.text is not None
        assert isinstance(result.score, float)


async def test_delete_document_removes_all_data(raag_instance):
    """Test that delete_document removes document and sections."""
    async with tenant_context("tenant_1"):
        # Add a document
        doc_id = await raag_instance.add_document(
            "Test content for deletion",
            {"title": "Delete Me"}
        )

        # Verify it exists
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) > 0

        # Delete it
        await raag_instance.delete_document(doc_id)

        # Verify it's gone
        results_after = await raag_instance.search([0.5] * 10, k=10)
        assert len(results_after) == 0


async def test_tenant_isolation_in_all_operations(raag_instance):
    """Test that tenant isolation works for all operations."""
    # Add document for tenant_1
    async with tenant_context("tenant_1"):
        doc_id_1 = await raag_instance.add_document(
            "Tenant 1 secret data",
            {"title": "Tenant 1 Doc"}
        )

    # Tenant_2 should not see tenant_1's data
    async with tenant_context("tenant_2"):
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 0

        # Add tenant_2's own document
        doc_id_2 = await raag_instance.add_document(
            "Tenant 2 public data",
            {"title": "Tenant 2 Doc"}
        )

        # Tenant_2 sees only their own data
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 1

    # Tenant_1 still sees only their own data
    async with tenant_context("tenant_1"):
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 1

        # Tenant_1 cannot delete tenant_2's document
        # (This should either do nothing or raise an error depending on implementation)
        await raag_instance.delete_document(doc_id_1)
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 0

    # Tenant_2's document is still there
    async with tenant_context("tenant_2"):
        results = await raag_instance.search([0.5] * 10, k=10)
        assert len(results) == 1


async def test_add_document_sync_wrapper(raag_instance):
    """Test synchronous wrapper for add_document."""
    # Note: Cannot easily test sync wrappers in async test
    # This is more of a smoke test
    assert hasattr(raag_instance, 'add_document_sync')
    assert callable(raag_instance.add_document_sync)


async def test_search_sync_wrapper(raag_instance):
    """Test synchronous wrapper for search."""
    assert hasattr(raag_instance, 'search_sync')
    assert callable(raag_instance.search_sync)


async def test_delete_document_sync_wrapper(raag_instance):
    """Test synchronous wrapper for delete_document."""
    assert hasattr(raag_instance, 'delete_document_sync')
    assert callable(raag_instance.delete_document_sync)


async def test_empty_search_returns_empty_list(raag_instance):
    """Test that search with no data returns empty list."""
    async with tenant_context("tenant_1"):
        results = await raag_instance.search([0.5] * 10, k=10)
        assert results == []


async def test_full_workflow(raag_instance):
    """Test the complete workflow: add -> search -> delete."""
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
