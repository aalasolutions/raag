"""CRUD operations with tenant filtering."""

from sqlalchemy import select, delete
from sqlalchemy.schema import CreateTable, CreateIndex
from raag.storage.models import Base, DocumentModel, SectionModel
from raag.storage.connection import create_session_factory
from raag.tenant.context import require_tenant
from raag.models.document import Document
from raag.models.section import Section


async def init_tables(engine):
    """Create all tables in the database.

    Safe to call concurrently: every statement uses IF NOT EXISTS.

    Args:
        engine: SQLAlchemy async engine
    """
    async with engine.begin() as conn:
        await conn.run_sync(_create_all_if_not_exists)


def _create_all_if_not_exists(sync_conn):
    for table in Base.metadata.sorted_tables:
        sync_conn.execute(CreateTable(table, if_not_exists=True))
        for index in table.indexes:
            sync_conn.execute(CreateIndex(index, if_not_exists=True))


async def save_document(session, doc: Document) -> str:
    """Save a document to the database.

    Args:
        session: SQLAlchemy async session
        doc: Document model to save

    Returns:
        The document ID

    Raises:
        RuntimeError: If no tenant context is set
        ValueError: If doc.tenant_id does not match the current tenant
    """
    tenant_id = require_tenant()
    if doc.tenant_id != tenant_id:
        raise ValueError(
            f"Document tenant_id '{doc.tenant_id}' does not match current tenant '{tenant_id}'"
        )
    model = DocumentModel(
        tenant_id=doc.tenant_id,
        doc_id=doc.doc_id,
        title=doc.title,
        meta=doc.metadata,
        created_at=doc.created_at
    )
    session.add(model)
    await session.commit()
    return model.doc_id


async def save_sections(session, sections: list[Section]) -> list[str]:
    """Save multiple sections to the database.

    Args:
        session: SQLAlchemy async session
        sections: List of Section models to save

    Returns:
        List of section IDs

    Raises:
        RuntimeError: If no tenant context is set
        ValueError: If any section's tenant_id does not match the current tenant,
            or its doc_id is not a document of the current tenant
    """
    tenant_id = require_tenant()
    for s in sections:
        if s.tenant_id != tenant_id:
            raise ValueError(
                f"Section tenant_id '{s.tenant_id}' does not match current tenant '{tenant_id}'"
            )
    doc_ids = {s.doc_id for s in sections}
    if doc_ids:
        result = await session.execute(
            select(DocumentModel.doc_id).where(
                DocumentModel.doc_id.in_(doc_ids),
                DocumentModel.tenant_id == tenant_id,
            )
        )
        missing = doc_ids - set(result.scalars().all())
        if missing:
            raise ValueError(
                f"Document '{sorted(missing)[0]}' does not exist or does not belong "
                f"to current tenant '{tenant_id}'"
            )
    models = [
        SectionModel(
            tenant_id=s.tenant_id,
            section_id=s.section_id,
            doc_id=s.doc_id,
            text=s.text,
            meta=s.metadata,
            created_at=s.created_at
        )
        for s in sections
    ]
    session.add_all(models)
    await session.commit()
    return [m.section_id for m in models]


async def get_sections_by_doc(session, doc_id: str) -> list[Section]:
    """Get all sections for a document, filtered by current tenant.

    Args:
        session: SQLAlchemy async session
        doc_id: Document ID to fetch sections for

    Returns:
        List of Section models

    Raises:
        RuntimeError: If no tenant context is set
    """
    tenant_id = require_tenant()
    result = await session.execute(
        select(SectionModel).where(
            SectionModel.doc_id == doc_id,
            SectionModel.tenant_id == tenant_id
        )
    )
    models = result.scalars().all()
    return [
        Section(
            tenant_id=m.tenant_id,
            section_id=m.section_id,
            doc_id=m.doc_id,
            text=m.text,
            metadata=m.meta,
            created_at=m.created_at
        )
        for m in models
    ]


async def list_documents(session) -> list[Document]:
    """List all documents for the current tenant.

    Args:
        session: SQLAlchemy async session

    Returns:
        List of Document models, oldest first

    Raises:
        RuntimeError: If no tenant context is set
    """
    tenant_id = require_tenant()
    result = await session.execute(
        select(DocumentModel)
        .where(DocumentModel.tenant_id == tenant_id)
        .order_by(DocumentModel.created_at)
    )
    models = result.scalars().all()
    return [
        Document(
            tenant_id=m.tenant_id,
            doc_id=m.doc_id,
            title=m.title,
            metadata=m.meta,
            created_at=m.created_at
        )
        for m in models
    ]


async def delete_document(session, doc_id: str) -> None:
    """Delete a document and all its sections, filtered by current tenant.

    Args:
        session: SQLAlchemy async session
        doc_id: Document ID to delete

    Raises:
        RuntimeError: If no tenant context is set
    """
    tenant_id = require_tenant()
    # Cascade delete sections first
    await session.execute(
        delete(SectionModel).where(
            SectionModel.doc_id == doc_id,
            SectionModel.tenant_id == tenant_id
        )
    )
    await session.execute(
        delete(DocumentModel).where(
            DocumentModel.doc_id == doc_id,
            DocumentModel.tenant_id == tenant_id
        )
    )
    await session.commit()
