"""Unit tests for storage layer with tenant isolation."""

import pytest
from datetime import datetime
from raag.storage.connection import create_engine, create_session_factory
from raag.storage.crud import init_tables, save_document, save_sections, get_sections_by_doc, delete_document
from raag.storage.models import DocumentModel, SectionModel
from raag.models.document import Document
from raag.models.section import Section
from raag.tenant.context import tenant_context, get_current_tenant, require_tenant


@pytest.fixture
async def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(":memory:")
    await init_tables(engine)
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        yield session
    await engine.dispose()


class TestTenantContext:
    """Tests for tenant context management."""

    @pytest.mark.asyncio
    async def test_get_current_tenant_returns_none_when_not_set(self):
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_tenant_context_sets_and_clears_tenant(self):
        assert get_current_tenant() is None
        async with tenant_context("tenant_1"):
            assert get_current_tenant() == "tenant_1"
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_nested_contexts_work_correctly(self):
        async with tenant_context("outer"):
            assert get_current_tenant() == "outer"
            async with tenant_context("inner"):
                assert get_current_tenant() == "inner"
            assert get_current_tenant() == "outer"
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_require_tenant_raises_when_not_set(self):
        with pytest.raises(RuntimeError, match="No tenant context set"):
            require_tenant()

    @pytest.mark.asyncio
    async def test_require_tenant_returns_value_when_set(self):
        async with tenant_context("tenant_1"):
            assert require_tenant() == "tenant_1"


class TestDocumentStorage:
    """Tests for document CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_document(self, in_memory_db):
        doc = Document(
            tenant_id="tenant_1",
            title="Test Document",
            metadata={"key": "value"}
        )
        doc_id = await save_document(in_memory_db, doc)
        assert doc_id == doc.doc_id

        # Verify it was saved
        from sqlalchemy import select
        result = await in_memory_db.execute(
            select(DocumentModel).where(DocumentModel.doc_id == doc_id)
        )
        model = result.scalar_one()
        assert model.title == "Test Document"
        assert model.tenant_id == "tenant_1"

    @pytest.mark.asyncio
    async def test_save_sections(self, in_memory_db):
        sections = [
            Section(
                tenant_id="tenant_1",
                doc_id="doc_1",
                text="Section 1",
                metadata={"index": 0}
            ),
            Section(
                tenant_id="tenant_1",
                doc_id="doc_1",
                text="Section 2",
                metadata={"index": 1}
            ),
        ]
        section_ids = await save_sections(in_memory_db, sections)
        assert len(section_ids) == 2
        assert section_ids[0] == sections[0].section_id

    @pytest.mark.asyncio
    async def test_get_sections_by_doc(self, in_memory_db):
        async with tenant_context("tenant_1"):
            sections = [
                Section(
                    tenant_id="tenant_1",
                    doc_id="doc_1",
                    text=f"Section {i}",
                    metadata={"index": i}
                )
                for i in range(3)
            ]
            await save_sections(in_memory_db, sections)

            retrieved = await get_sections_by_doc(in_memory_db, "doc_1")
            assert len(retrieved) == 3
            assert retrieved[0].text == "Section 0"

    @pytest.mark.asyncio
    async def test_delete_document_cascades_to_sections(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc = Document(tenant_id="tenant_1", title="To Delete")
            await save_document(in_memory_db, doc)

            sections = [
                Section(
                    tenant_id="tenant_1",
                    doc_id=doc.doc_id,
                    text=f"Section {i}",
                    metadata={"index": i}
                )
                for i in range(2)
            ]
            await save_sections(in_memory_db, sections)

            await delete_document(in_memory_db, doc.doc_id)

            # Verify document is gone
            from sqlalchemy import select
            result = await in_memory_db.execute(
                select(DocumentModel).where(DocumentModel.doc_id == doc.doc_id)
            )
            assert result.scalar_one_or_none() is None

            # Verify sections are gone
            result = await in_memory_db.execute(
                select(SectionModel).where(SectionModel.doc_id == doc.doc_id)
            )
            assert list(result.scalars().all()) == []


class TestTenantIsolation:
    """Tests that verify tenant isolation works correctly."""

    @pytest.mark.asyncio
    async def test_get_sections_filters_by_tenant(self, in_memory_db):
        # Create sections for tenant_1 (unique doc_id per tenant)
        async with tenant_context("tenant_1"):
            sections = [
                Section(
                    tenant_id="tenant_1",
                    doc_id="doc_tenant_1",
                    text="Tenant 1 Section",
                    metadata={}
                )
            ]
            await save_sections(in_memory_db, sections)

        # Create sections for tenant_2 (different doc_id)
        async with tenant_context("tenant_2"):
            sections = [
                Section(
                    tenant_id="tenant_2",
                    doc_id="doc_tenant_2",
                    text="Tenant 2 Section",
                    metadata={}
                )
            ]
            await save_sections(in_memory_db, sections)

        # Tenant 1 only sees their own sections
        async with tenant_context("tenant_1"):
            retrieved = await get_sections_by_doc(in_memory_db, "doc_tenant_1")
            assert len(retrieved) == 1
            assert retrieved[0].text == "Tenant 1 Section"

        # Tenant 2 only sees their own sections
        async with tenant_context("tenant_2"):
            retrieved = await get_sections_by_doc(in_memory_db, "doc_tenant_2")
            assert len(retrieved) == 1
            assert retrieved[0].text == "Tenant 2 Section"

    @pytest.mark.asyncio
    async def test_delete_document_only_affects_current_tenant(self, in_memory_db):
        # Create documents for both tenants (auto-generated doc_ids)
        async with tenant_context("tenant_1"):
            doc1 = Document(tenant_id="tenant_1", title="Tenant 1 Doc")
            doc1_id = await save_document(in_memory_db, doc1)

        async with tenant_context("tenant_2"):
            doc2 = Document(tenant_id="tenant_2", title="Tenant 2 Doc")
            doc2_id = await save_document(in_memory_db, doc2)

        # Delete from tenant_1
        async with tenant_context("tenant_1"):
            await delete_document(in_memory_db, doc1_id)

        # Verify tenant_1's document is gone
        from sqlalchemy import select
        async with tenant_context("tenant_1"):
            result = await in_memory_db.execute(
                select(DocumentModel).where(
                    DocumentModel.doc_id == doc1_id,
                    DocumentModel.tenant_id == "tenant_1"
                )
            )
            assert result.scalar_one_or_none() is None

        # Verify tenant_2's document still exists
        async with tenant_context("tenant_2"):
            result = await in_memory_db.execute(
                select(DocumentModel).where(
                    DocumentModel.doc_id == doc2_id,
                    DocumentModel.tenant_id == "tenant_2"
                )
            )
            assert result.scalar_one() is not None
