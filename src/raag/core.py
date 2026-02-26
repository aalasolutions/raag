"""Core Raag class for RAG operations."""

import asyncio
from typing import Callable, Awaitable

from raag.config import RaagConfig
from raag.protocols import VectorStore, VectorItem
from raag.vector.memory import InMemoryVectorStore
from raag.splitters.base import Splitter
from raag.splitters.markdown import MarkdownSplitter
from raag.storage.connection import create_engine, create_session_factory
from raag.storage.crud import init_tables, save_document, save_sections, get_sections_by_doc, delete_document as db_delete_document
from raag.storage.models import SectionModel
from raag.models.document import Document
from raag.models.section import Section
from raag.models.results import SearchResult
from raag.tenant.context import require_tenant

from sqlalchemy import select


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

    async def _ensure_tables(self):
        """Ensure database tables are created."""
        if not self._initialized:
            await init_tables(self._engine)
            self._initialized = True

    async def add_document(self, text: str, metadata: dict) -> str:
        """Add a document to the RAAG system.

        Splits the text into sections (by double newline), generates
        embeddings, and stores in both database and vector store.

        Args:
            text: Document text content.
            metadata: Optional metadata (e.g., {"title": "My Document"}).

        Returns:
            The document ID.

        Raises:
            RuntimeError: If no tenant context is set.
        """
        await self._ensure_tables()
        tenant_id = require_tenant()

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

        # Generate embeddings
        embeddings = await self.embed_fn([s.text for s in sections])

        # Save to DB
        async with self._session_factory() as session:
            await save_document(session, doc)
            await save_sections(session, sections)

        # Save to vector store
        vector_items = [
            VectorItem(
                id=s.section_id,
                vector=emb,
                metadata={"doc_id": s.doc_id},
                tenant_id=tenant_id,
            )
            for s, emb in zip(sections, embeddings)
        ]
        await self.vector_store.upsert(vector_items)

        return doc.doc_id

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
        await self._ensure_tables()
        tenant_id = require_tenant()
        k = k if k is not None else self.config.default_k

        # Search vector store
        vector_results = await self.vector_store.search(query_embedding, k, tenant_id)

        if not vector_results:
            return []

        # Get section texts from DB (tenant filter for defense in depth)
        section_ids = [r.id for r in vector_results]
        async with self._session_factory() as session:
            result = await session.execute(
                select(SectionModel).where(
                    SectionModel.section_id.in_(section_ids),
                    SectionModel.tenant_id == tenant_id,
                )
            )
            sections = {s.section_id: s for s in result.scalars().all()}

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
        await self._ensure_tables()
        tenant_id = require_tenant()

        # Get sections before delete (for vector store cleanup)
        async with self._session_factory() as session:
            sections = await get_sections_by_doc(session, doc_id)
            section_ids = [s.section_id for s in sections]
            await db_delete_document(session, doc_id)

        # Delete from vector store
        await self.vector_store.delete(section_ids, tenant_id)

    # Sync wrappers

    def _run_sync(self, coro):
        """Run an async coroutine synchronously.

        Uses asyncio.run() when no loop is running.
        Falls back to a new thread when called from inside an existing
        event loop (e.g., FastAPI, Django async views, Jupyter notebooks).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def add_document_sync(self, text: str, metadata: dict) -> str:
        """Synchronous wrapper for add_document."""
        return self._run_sync(self.add_document(text, metadata))

    def search_sync(self, query_embedding: list[float], k: int | None = None) -> list[SearchResult]:
        """Synchronous wrapper for search."""
        return self._run_sync(self.search(query_embedding, k))

    def delete_document_sync(self, doc_id: str) -> None:
        """Synchronous wrapper for delete_document."""
        return self._run_sync(self.delete_document(doc_id))
