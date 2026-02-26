"""CRUD operations with tenant filtering."""

from sqlalchemy import select, delete
from raag.storage.models import Base, DocumentModel, SectionModel
from raag.storage.connection import create_session_factory
from raag.tenant.context import require_tenant
from raag.models.document import Document
from raag.models.section import Section


async def init_tables(engine):
    """Create all tables in the database.

    Args:
        engine: SQLAlchemy async engine
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_document(session, doc: Document) -> str:
    """Save a document to the database.

    Args:
        session: SQLAlchemy async session
        doc: Document model to save

    Returns:
        The document ID
    """
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
    """
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
