"""SQLAlchemy table definitions."""

from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone
from uuid import uuid4


class Base(DeclarativeBase):
    """Base class for all RAAG ORM models."""


class DocumentModel(Base):
    """SQLAlchemy model for documents."""

    __tablename__ = "__raag_documents"

    tenant_id = Column(String, nullable=False, index=True)
    doc_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    title = Column(String, nullable=False)
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SectionModel(Base):
    """SQLAlchemy model for document sections."""

    __tablename__ = "__raag_sections"

    tenant_id = Column(String, nullable=False, index=True)
    section_id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    doc_id = Column(String, ForeignKey("__raag_documents.doc_id"), nullable=False, index=True)
    text = Column(Text, nullable=False)
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
