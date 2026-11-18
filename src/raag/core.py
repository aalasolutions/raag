"""Core Raag class for RAG operations."""

import asyncio
import contextlib
import contextvars
import threading
import weakref
from typing import Callable, Awaitable

from raag.config import RaagConfig
from raag.protocols import VectorStore, VectorItem
from raag.vector.memory import InMemoryVectorStore
from raag.splitters.base import Splitter
from raag.splitters.markdown import MarkdownSplitter
from raag.storage.connection import create_engine, create_session_factory
from raag.storage.crud import init_tables, save_document, save_sections, get_sections_by_doc, list_documents as db_list_documents, delete_document as db_delete_document
from raag.storage.models import SectionModel
from raag.models.document import Document
from raag.models.section import Section
from raag.models.results import SearchResult
from raag.tenant.context import require_tenant

from sqlalchemy import select
from sqlalchemy.pool import StaticPool


class Raag:
    """Main RAAG class for document storage and retrieval.

    Provides a simple API for adding, searching, and deleting documents
    with automatic multi-tenant isolation.

    Example:
        async def demo():
            raag = Raag(embed_fn=my_embed, db_path="test.db")

            async with tenant_context("client_1"):
                doc_id = await raag.add_document("Hello world", {"title": "Test"})
                results = await raag.search(query_embedding, k=5)
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
        db_path: str = "raag.db",
        vector_store: VectorStore | None = None,
        config: RaagConfig | None = None,
        splitter: Splitter | None = None,
    ):
        """Initialize the Raag instance.

        Args:
            embed_fn: Async function that takes a list of texts and returns
                     a list of embedding vectors (list of floats).
            db_path: Path to SQLite database file.
            vector_store: Optional vector store implementation. If None,
                         uses InMemoryVectorStore.
            config: Optional configuration. If None, uses defaults.
            splitter: Optional document splitter. If None,
                     uses MarkdownSplitter.
        """
        self.embed_fn = embed_fn
        self.config = config or RaagConfig()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.splitter = splitter or MarkdownSplitter()
        self._engine = create_engine(db_path)
        self._session_factory = create_session_factory(self._engine)
        self._initialized = False
        self._db_loop = None
        self._db_loop_lock = threading.Lock()
        # One shared connection (":memory:") means one shared transaction
        self._session_lock = asyncio.Lock() if isinstance(self._engine.pool, StaticPool) else None

    def _get_db_loop(self) -> asyncio.AbstractEventLoop:
        """Return the event loop that owns the engine, starting it on first use.

        The engine's pool, locks and aiosqlite connections are bound to one
        loop, so every database call runs here, whatever loop or thread the
        caller is on.
        """
        with self._db_loop_lock:
            if self._db_loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=_run_loop, args=(loop,), name="raag-db", daemon=True).start()
                finalizer = weakref.finalize(self, _stop_loop, loop, self._engine)
                finalizer.atexit = False
                self._db_loop = loop
            return self._db_loop

    async def _db(self, fn):
        """Run fn(session) on the database loop and return its result.

        The caller's context, including the tenant, is carried over.
        """
        async def run():
            async with self._session_lock or contextlib.nullcontext():
                if not self._initialized:
                    await init_tables(self._engine)
                    self._initialized = True
                async with self._session_factory() as session:
                    return await fn(session)

        future = asyncio.run_coroutine_threadsafe(run(), self._get_db_loop())
        return await asyncio.wrap_future(future)

    async def add_document(self, text: str, metadata: dict) -> str:
        """Add a document to the RAAG system.

        Splits the text into sections with the configured splitter (Markdown
        headings by default), embeds them in batches of
        config.embed_batch_size, and stores them in both database and
        vector store.

        Args:
            text: Document text content.
            metadata: Document metadata dict, may be empty. Its "title" key
                sets the title (default "Untitled") and "source_path" is
                copied to each section.

        Returns:
            The document ID.

        Raises:
            RuntimeError: If no tenant context is set.
            ValueError: If config.embed_batch_size is less than 1, or
                embed_fn returns a different number of vectors than texts.

        If storing fails, the document's rows and vectors are removed
        before the error is raised.
        """
        tenant_id = require_tenant()
        batch_size = self.config.embed_batch_size
        if batch_size < 1:
            raise ValueError(f"embed_batch_size must be at least 1, got {batch_size}")

        # Create document
        doc = Document(
            tenant_id=tenant_id,
            title=metadata.get("title", "Untitled"),
            metadata=metadata,
        )

        # Split text into sections using configured splitter
        split_sections = self.splitter.split(text, self.config.max_section_chars)

        # Create sections with splitter metadata
        source_path = metadata.get("source_path", "")
        sections = [
            Section(
                tenant_id=tenant_id,
                doc_id=doc.doc_id,
                text=ss.text,
                metadata={
                    "heading_level": ss.heading_level,
                    "section_number": ss.section_number,
                    "section_title": ss.section_title,
                    "parent_path": ss.parent_path,
                    "chunk_index": ss.chunk_index,
                    "total_chunks": ss.total_chunks,
                    "source_path": source_path,
                },
            )
            for ss in split_sections
        ]

        # Generate embeddings in batches
        texts = [s.text for s in sections]
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors = await self.embed_fn(batch)
            if len(vectors) != len(batch):
                raise ValueError(
                    f"embed_fn returned {len(vectors)} vectors for {len(batch)} texts"
                )
            embeddings.extend(vectors)

        vector_items = [
            VectorItem(
                id=s.section_id,
                vector=emb,
                metadata={"doc_id": s.doc_id},
                tenant_id=tenant_id,
            )
            for s, emb in zip(sections, embeddings)
        ]

        async def save(session):
            await save_document(session, doc)
            await save_sections(session, sections)

        try:
            await self._db(save)
            if vector_items:
                await self.vector_store.upsert(vector_items)
        except BaseException:
            await _run_to_completion(
                self._discard(doc.doc_id, [s.section_id for s in sections], tenant_id)
            )
            raise

        return doc.doc_id

    async def _discard(self, doc_id: str, section_ids: list[str], tenant_id: str) -> None:
        """Remove what a failed add_document stored. Cleanup errors are ignored."""
        try:
            await self._db(lambda session: db_delete_document(session, doc_id))
        except Exception:
            pass
        try:
            await self.vector_store.delete(section_ids, tenant_id)
        except Exception:
            pass

    async def search(self, query_embedding: list[float], k: int | None = None) -> list[SearchResult]:
        """Search for similar document sections.

        Args:
            query_embedding: The query vector to search with.
            k: Number of results to return. Defaults to config.default_k.

        Returns:
            List of search results, sorted by relevance score (descending).

        Raises:
            RuntimeError: If no tenant context is set.
        """
        tenant_id = require_tenant()
        k = k if k is not None else self.config.default_k

        # Search vector store
        vector_results = await self.vector_store.search(query_embedding, k, tenant_id)

        if not vector_results:
            return []

        # Get section texts from DB (tenant filter for defense in depth)
        section_ids = [r.id for r in vector_results]

        async def load(session):
            result = await session.execute(
                select(SectionModel).where(
                    SectionModel.section_id.in_(section_ids),
                    SectionModel.tenant_id == tenant_id,
                )
            )
            return {s.section_id: s for s in result.scalars().all()}

        sections = await self._db(load)

        # Build results
        results = []
        for vr in vector_results:
            section = sections.get(vr.id)
            if section:
                results.append(SearchResult(
                    section_id=vr.id,
                    doc_id=section.doc_id,
                    text=section.text,
                    score=vr.score,
                    metadata=vr.metadata,
                ))

        return results

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its sections.

        Args:
            doc_id: Document ID to delete.

        Raises:
            RuntimeError: If no tenant context is set.
        """
        tenant_id = require_tenant()

        # Get sections before delete (for vector store cleanup)
        async def remove(session):
            sections = await get_sections_by_doc(session, doc_id)
            await db_delete_document(session, doc_id)
            return [s.section_id for s in sections]

        section_ids = await self._db(remove)

        # Delete from vector store
        await self.vector_store.delete(section_ids, tenant_id)

    async def list_documents(self) -> list[Document]:
        """List all documents for the current tenant.

        Returns:
            List of documents, oldest first.

        Raises:
            RuntimeError: If no tenant context is set.
        """
        require_tenant()
        return await self._db(db_list_documents)

    # Sync wrappers

    def _run_sync(self, coro):
        """Run an async coroutine synchronously.

        Uses asyncio.run() when no loop is running.
        Falls back to a new thread when called from inside an existing
        event loop (e.g., FastAPI, Django async views, Jupyter notebooks).
        The caller's context, including the tenant, is carried into the thread.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(ctx.run, asyncio.run, coro).result()

    def add_document_sync(self, text: str, metadata: dict) -> str:
        """Synchronous wrapper for add_document."""
        return self._run_sync(self.add_document(text, metadata))

    def search_sync(self, query_embedding: list[float], k: int | None = None) -> list[SearchResult]:
        """Synchronous wrapper for search."""
        return self._run_sync(self.search(query_embedding, k))

    def delete_document_sync(self, doc_id: str) -> None:
        """Synchronous wrapper for delete_document."""
        return self._run_sync(self.delete_document(doc_id))


async def _run_to_completion(coro) -> None:
    """Await coro to the end even if the caller is cancelled meanwhile.

    A cancellation received while waiting is raised once coro has finished.
    """
    task = asyncio.ensure_future(coro)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                cancelled = True
    if cancelled:
        raise asyncio.CancelledError


def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()


def _stop_loop(loop: asyncio.AbstractEventLoop, engine) -> None:
    """Dispose the engine on its loop, then stop the loop. Safe from any thread."""
    async def close():
        try:
            await engine.dispose()
        finally:
            loop.stop()

    coro = close()
    try:
        asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        coro.close()
