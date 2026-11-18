"""Unit tests for storage layer with tenant isolation."""

import asyncio
import contextvars
import gc
import sys
import threading
import pytest
from datetime import datetime
from sqlalchemy import select
from raag.storage.connection import create_engine, create_session_factory
from raag.storage.crud import init_tables, save_document, save_sections, get_sections_by_doc, list_documents, delete_document
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


async def _all_documents(session):
    result = await session.execute(select(DocumentModel))
    return list(result.scalars().all())


async def _all_sections(session):
    result = await session.execute(select(SectionModel))
    return list(result.scalars().all())


async def _saved_document(session, tenant_id, title="Doc"):
    doc = Document(tenant_id=tenant_id, title=title)
    await save_document(session, doc)
    return doc.doc_id


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

    def test_sync_with_sets_and_clears_tenant(self):
        assert get_current_tenant() is None
        with tenant_context("tenant_1"):
            assert get_current_tenant() == "tenant_1"
            with tenant_context("inner"):
                assert get_current_tenant() == "inner"
            assert get_current_tenant() == "tenant_1"
        assert get_current_tenant() is None

    def test_sync_with_clears_tenant_on_exception(self):
        with pytest.raises(ValueError):
            with tenant_context("tenant_1"):
                raise ValueError("boom")
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self):
        @tenant_context("tenant_1")
        async def handler(value):
            """Handler docstring."""
            await asyncio.sleep(0)
            return get_current_tenant(), value

        assert handler.__name__ == "handler"
        assert handler.__doc__ == "Handler docstring."
        assert await handler(1) == ("tenant_1", 1)
        assert await handler(2) == ("tenant_1", 2)
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_decorator_concurrent_tasks_and_nesting(self):
        def make(tenant_id):
            @tenant_context(tenant_id)
            async def handler():
                seen = []
                for _ in range(3):
                    seen.append(get_current_tenant())
                    await asyncio.sleep(0)
                async with tenant_context("inner"):
                    seen.append(get_current_tenant())
                seen.append(get_current_tenant())
                return seen
            return handler

        shared = make("shared")
        handlers = [make(f"tenant_{i}") for i in range(5)] + [shared, shared]
        results = await asyncio.gather(*(h() for h in handlers))
        expected = [f"tenant_{i}" for i in range(5)] + ["shared", "shared"]
        for tenant_id, seen in zip(expected, results):
            assert seen == [tenant_id] * 3 + ["inner", tenant_id]
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_decorator_clears_tenant_on_exception(self):
        @tenant_context("tenant_1")
        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await failing()
        assert get_current_tenant() is None

    def test_decorator_on_sync_function(self):
        @tenant_context("tenant_1")
        def handler():
            return get_current_tenant()

        assert handler() == "tenant_1"
        with tenant_context("outer"):
            assert handler() == "tenant_1"
            assert get_current_tenant() == "outer"
        assert get_current_tenant() is None


    def test_decorator_rejects_generators(self):
        def numbers():
            yield get_current_tenant()

        async def async_numbers():
            yield get_current_tenant()

        for func in (numbers, async_numbers):
            with pytest.raises(TypeError, match="cannot decorate a generator"):
                tenant_context("tenant_1")(func)

    def test_collected_generator_leaves_current_context_alone(self, monkeypatch):
        errors = []
        monkeypatch.setattr(sys, "unraisablehook", lambda u: errors.append(u.exc_value))

        def holds_context():
            with tenant_context("generator"):
                yield

        class Holder:
            pass

        holder = Holder()
        holder.cycle = holder
        holder.gen = holds_context()
        contextvars.Context().run(next, holder.gen)
        del holder

        with tenant_context("bystander"):
            gc.collect()
            assert get_current_tenant() == "bystander"
        assert get_current_tenant() is None
        assert [str(e) for e in errors] == [
            "tenant_context exited out of order or in a different context than it was entered"
        ]

    @pytest.mark.asyncio
    async def test_one_context_object_cannot_overlap_in_tasks(self):
        ctx = tenant_context("shared")
        started = asyncio.Event()

        async def first():
            async with ctx:
                started.set()
                await asyncio.sleep(0.01)
                return get_current_tenant()

        async def second():
            await started.wait()
            with pytest.raises(RuntimeError, match="already active"):
                async with ctx:
                    pass
            return get_current_tenant()

        assert await asyncio.gather(first(), second()) == ["shared", None]
        async with ctx:
            assert get_current_tenant() == "shared"
        assert get_current_tenant() is None

    def test_one_context_object_cannot_overlap_in_threads(self):
        ctx = tenant_context("shared")
        entered, release = threading.Event(), threading.Event()
        errors = []

        def hold():
            with ctx:
                entered.set()
                release.wait(10)

        thread = threading.Thread(target=hold)
        thread.start()
        assert entered.wait(10)
        try:
            ctx.__enter__()
        except RuntimeError as exc:
            errors.append(str(exc))
        release.set()
        thread.join(10)
        assert len(errors) == 1 and "already active" in errors[0]
        assert get_current_tenant() is None

    def test_one_context_object_nested_in_itself_raises(self):
        ctx = tenant_context("tenant_1")
        with ctx:
            with tenant_context("inner"):
                with pytest.raises(RuntimeError, match="already active"):
                    with ctx:
                        pass
                assert get_current_tenant() == "inner"
            assert get_current_tenant() == "tenant_1"
        assert get_current_tenant() is None
        with ctx:
            assert get_current_tenant() == "tenant_1"
        assert get_current_tenant() is None

    @pytest.mark.asyncio
    async def test_exit_in_child_task_leaves_both_contexts_alone(self):
        ctx = tenant_context("tenant_1")

        async def child():
            with pytest.raises(RuntimeError, match="different context"):
                ctx.__exit__(None, None, None)
            return get_current_tenant()

        async with ctx:
            assert await asyncio.create_task(child()) == "tenant_1"
            assert get_current_tenant() == "tenant_1"
        assert get_current_tenant() is None

    def test_exit_out_of_order_raises_clear_error(self):
        outer = tenant_context("outer")
        inner = tenant_context("inner")
        outer.__enter__()
        inner.__enter__()
        with pytest.raises(RuntimeError, match="out of order"):
            outer.__exit__(None, None, None)
        inner.__exit__(None, None, None)
        outer.__exit__(None, None, None)
        assert get_current_tenant() is None


class TestDocumentStorage:
    """Tests for document CRUD operations."""

    @pytest.mark.asyncio
    async def test_save_document(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc = Document(
                tenant_id="tenant_1",
                title="Test Document",
                metadata={"key": "value"}
            )
            doc_id = await save_document(in_memory_db, doc)
        assert doc_id == doc.doc_id

        result = await in_memory_db.execute(
            select(DocumentModel).where(DocumentModel.doc_id == doc_id)
        )
        model = result.scalar_one()
        assert model.title == "Test Document"
        assert model.tenant_id == "tenant_1"

    @pytest.mark.asyncio
    async def test_save_document_with_tenant_isolation(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc1 = Document(tenant_id="tenant_1", title="Tenant 1 Doc")
            await save_document(in_memory_db, doc1)

        async with tenant_context("tenant_2"):
            doc2 = Document(tenant_id="tenant_2", title="Tenant 2 Doc")
            await save_document(in_memory_db, doc2)

            # Writing tenant_1's document while tenant_2 is current fails
            foreign = Document(tenant_id="tenant_1", title="Foreign Doc")
            with pytest.raises(ValueError, match="does not match current tenant"):
                await save_document(in_memory_db, foreign)

        rows = {m.doc_id: m.tenant_id for m in await _all_documents(in_memory_db)}
        assert rows == {doc1.doc_id: "tenant_1", doc2.doc_id: "tenant_2"}

        async with tenant_context("tenant_1"):
            assert [d.doc_id for d in await list_documents(in_memory_db)] == [doc1.doc_id]
        async with tenant_context("tenant_2"):
            assert [d.doc_id for d in await list_documents(in_memory_db)] == [doc2.doc_id]

    @pytest.mark.asyncio
    async def test_save_document_without_tenant_context_fails(self, in_memory_db):
        doc = Document(tenant_id="tenant_1", title="No Context")
        with pytest.raises(RuntimeError, match="No tenant context set"):
            await save_document(in_memory_db, doc)
        assert await _all_documents(in_memory_db) == []

    @pytest.mark.asyncio
    async def test_save_sections(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc_id = await _saved_document(in_memory_db, "tenant_1")
            sections = [
                Section(
                    tenant_id="tenant_1",
                    doc_id=doc_id,
                    text="Section 1",
                    metadata={"index": 0}
                ),
                Section(
                    tenant_id="tenant_1",
                    doc_id=doc_id,
                    text="Section 2",
                    metadata={"index": 1}
                ),
            ]
            section_ids = await save_sections(in_memory_db, sections)
        assert section_ids == [s.section_id for s in sections]

    @pytest.mark.asyncio
    async def test_save_sections_with_tenant_isolation(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc_1 = await _saved_document(in_memory_db, "tenant_1")
            await save_sections(in_memory_db, [
                Section(tenant_id="tenant_1", doc_id=doc_1, text="Tenant 1 Section")
            ])

        async with tenant_context("tenant_2"):
            doc_2 = await _saved_document(in_memory_db, "tenant_2")
            await save_sections(in_memory_db, [
                Section(tenant_id="tenant_2", doc_id=doc_2, text="Tenant 2 Section")
            ])

            # One foreign section rejects the whole batch
            mixed = [
                Section(tenant_id="tenant_2", doc_id=doc_2, text="Own Section"),
                Section(tenant_id="tenant_1", doc_id=doc_1, text="Foreign Section"),
            ]
            with pytest.raises(ValueError, match="does not match current tenant"):
                await save_sections(in_memory_db, mixed)

        rows = sorted((m.tenant_id, m.text) for m in await _all_sections(in_memory_db))
        assert rows == [("tenant_1", "Tenant 1 Section"), ("tenant_2", "Tenant 2 Section")]

    @pytest.mark.asyncio
    async def test_save_sections_without_tenant_context_fails(self, in_memory_db):
        sections = [Section(tenant_id="tenant_1", doc_id="doc_1", text="No Context")]
        with pytest.raises(RuntimeError, match="No tenant context set"):
            await save_sections(in_memory_db, sections)
        assert await _all_sections(in_memory_db) == []

    @pytest.mark.asyncio
    async def test_get_sections_by_doc(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc_id = await _saved_document(in_memory_db, "tenant_1")
            sections = [
                Section(
                    tenant_id="tenant_1",
                    doc_id=doc_id,
                    text=f"Section {i}",
                    metadata={"index": i}
                )
                for i in range(3)
            ]
            await save_sections(in_memory_db, sections)

            retrieved = await get_sections_by_doc(in_memory_db, doc_id)
            assert len(retrieved) == 3
            assert retrieved[0].text == "Section 0"

    @pytest.mark.asyncio
    async def test_list_documents_returns_current_tenant_documents(self, in_memory_db):
        async with tenant_context("tenant_1"):
            second = Document(tenant_id="tenant_1", title="Second", created_at=datetime(2026, 1, 2))
            first = Document(
                tenant_id="tenant_1", title="First", metadata={"n": 1}, created_at=datetime(2026, 1, 1)
            )
            await save_document(in_memory_db, second)
            await save_document(in_memory_db, first)

            docs = await list_documents(in_memory_db)

        assert [d.doc_id for d in docs] == [first.doc_id, second.doc_id]
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].title == "First"
        assert docs[0].metadata == {"n": 1}
        assert docs[0].tenant_id == "tenant_1"

    @pytest.mark.asyncio
    async def test_list_documents_empty_and_requires_tenant(self, in_memory_db):
        async with tenant_context("tenant_1"):
            assert await list_documents(in_memory_db) == []
        with pytest.raises(RuntimeError, match="No tenant context set"):
            await list_documents(in_memory_db)

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
        async with tenant_context("tenant_1"):
            doc_1 = await _saved_document(in_memory_db, "tenant_1")
            await save_sections(in_memory_db, [
                Section(tenant_id="tenant_1", doc_id=doc_1, text="Tenant 1 Section")
            ])

        async with tenant_context("tenant_2"):
            doc_2 = await _saved_document(in_memory_db, "tenant_2")
            await save_sections(in_memory_db, [
                Section(tenant_id="tenant_2", doc_id=doc_2, text="Tenant 2 Section")
            ])

        async with tenant_context("tenant_1"):
            assert [s.text for s in await get_sections_by_doc(in_memory_db, doc_1)] == ["Tenant 1 Section"]
            assert await get_sections_by_doc(in_memory_db, doc_2) == []

        async with tenant_context("tenant_2"):
            assert [s.text for s in await get_sections_by_doc(in_memory_db, doc_2)] == ["Tenant 2 Section"]
            assert await get_sections_by_doc(in_memory_db, doc_1) == []

    @pytest.mark.asyncio
    async def test_tenant_cannot_query_other_tenants_data(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc = Document(tenant_id="tenant_1", title="Tenant 1 Doc")
            await save_document(in_memory_db, doc)
            await save_sections(in_memory_db, [
                Section(tenant_id="tenant_1", doc_id=doc.doc_id, text="Secret")
            ])

        async with tenant_context("tenant_2"):
            assert await get_sections_by_doc(in_memory_db, doc.doc_id) == []
            assert await list_documents(in_memory_db) == []

        async with tenant_context("tenant_1"):
            retrieved = await get_sections_by_doc(in_memory_db, doc.doc_id)
            assert [s.text for s in retrieved] == ["Secret"]

    @pytest.mark.asyncio
    async def test_delete_document_only_affects_current_tenant(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc1 = Document(tenant_id="tenant_1", title="Tenant 1 Doc")
            doc1_id = await save_document(in_memory_db, doc1)
            await save_sections(in_memory_db, [
                Section(tenant_id="tenant_1", doc_id=doc1_id, text="Tenant 1 Section")
            ])

        async with tenant_context("tenant_2"):
            doc2 = Document(tenant_id="tenant_2", title="Tenant 2 Doc")
            doc2_id = await save_document(in_memory_db, doc2)

            # Cross-tenant delete attempt is a no-op
            await delete_document(in_memory_db, doc1_id)

        docs = {m.doc_id for m in await _all_documents(in_memory_db)}
        assert docs == {doc1_id, doc2_id}
        assert [m.doc_id for m in await _all_sections(in_memory_db)] == [doc1_id]

        # Own delete removes only tenant_1's rows
        async with tenant_context("tenant_1"):
            await delete_document(in_memory_db, doc1_id)

        docs = {m.doc_id for m in await _all_documents(in_memory_db)}
        assert docs == {doc2_id}
        assert await _all_sections(in_memory_db) == []

    @pytest.mark.asyncio
    async def test_save_sections_rejects_other_tenants_document(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc = Document(tenant_id="tenant_1", title="Tenant 1 Doc")
            await save_document(in_memory_db, doc)

        async with tenant_context("tenant_2"):
            own = Document(tenant_id="tenant_2", title="Tenant 2 Doc")
            await save_document(in_memory_db, own)
            batch = [
                Section(tenant_id="tenant_2", doc_id=own.doc_id, text="Own"),
                Section(tenant_id="tenant_2", doc_id=doc.doc_id, text="Injected"),
            ]
            with pytest.raises(ValueError, match="does not belong to current tenant"):
                await save_sections(in_memory_db, batch)

            await save_sections(in_memory_db, batch[:1])

        rows = sorted((m.tenant_id, m.doc_id, m.text) for m in await _all_sections(in_memory_db))
        assert rows == [("tenant_2", own.doc_id, "Own")]

    @pytest.mark.asyncio
    async def test_save_sections_rejects_missing_document(self, in_memory_db):
        async with tenant_context("tenant_1"):
            doc_id = await _saved_document(in_memory_db, "tenant_1")
            batch = [
                Section(tenant_id="tenant_1", doc_id=doc_id, text="Own"),
                Section(tenant_id="tenant_1", doc_id="no_such_doc", text="Orphan"),
            ]
            with pytest.raises(ValueError, match="'no_such_doc' does not exist"):
                await save_sections(in_memory_db, batch)
        assert await _all_sections(in_memory_db) == []


class TestCreateEngine:
    """Tests for database URL handling."""

    @pytest.mark.parametrize("url", [
        "postgresql+asyncpg://user:secret@db.example.com/raag",
        "mysql+aiomysql://user:secret@localhost/raag",
    ])
    def test_non_sqlite_url_is_rejected(self, url):
        with pytest.raises(ValueError, match="Only SQLite is supported") as excinfo:
            create_engine(url)
        assert "secret" not in str(excinfo.value)

    @pytest.mark.parametrize("url", [
        "admin:hunter2@host://x",
        "postgresql://admin:hunter2@host://x",
        "sqlite+aiosqlite://admin:hunter2@host/x",
        "sqlite://admin:hunter2@host/x",
        "postgresql:admin:hunter2@host",
    ])
    def test_errors_never_echo_credentials(self, url):
        with pytest.raises(Exception) as excinfo:
            create_engine(url)
        assert "hunter2" not in str(excinfo.value)
        assert "hunter2" not in repr(excinfo.value)

    @pytest.mark.parametrize("db_path, database", [
        ("raag.db", "raag.db"),
        (":memory:", ":memory:"),
        ("sqlite_data.db", "sqlite_data.db"),
        ("sqlite+aiosqlite:///other.db", "other.db"),
        ("C:\\data\\raag.db", "C:\\data\\raag.db"),
        ("C:/data/raag.db", "C:/data/raag.db"),
        ("data/raag.db", "data/raag.db"),
        ("./notes:v2.db", "./notes:v2.db"),
    ])
    def test_sqlite_paths_and_urls(self, db_path, database):
        engine = create_engine(db_path)
        assert engine.url.drivername == "sqlite+aiosqlite"
        assert engine.url.database == database
        engine.sync_engine.dispose()

    @pytest.mark.parametrize("db_path", [
        "postgresql:user:pw@h",
        "postgresql+asyncpg:user:pw@h/raag",
        "mysql:/raag",
        "sqlite:raag.db",
    ])
    def test_url_without_slashes_is_rejected(self, db_path):
        with pytest.raises(ValueError, match="looks like a URL without '//'") as excinfo:
            create_engine(db_path)
        assert "pw" not in str(excinfo.value)

    async def test_memory_path_is_in_memory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        engine = create_engine(":memory:")
        async with engine.connect() as conn:
            rows = (await conn.exec_driver_sql("PRAGMA database_list")).all()
        await engine.dispose()
        assert [(r[1], r[2]) for r in rows] == [("main", "")]
        assert list(tmp_path.iterdir()) == []
