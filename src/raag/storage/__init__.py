"""SQLite storage layer with multi-tenant support."""

from raag.storage.connection import create_engine, create_session_factory
from raag.storage.models import Base, DocumentModel, SectionModel
from raag.storage.crud import (
    init_tables,
    save_document,
    save_sections,
    get_sections_by_doc,
    list_documents,
    delete_document,
)

__all__ = [
    "create_engine",
    "create_session_factory",
    "Base",
    "DocumentModel",
    "SectionModel",
    "init_tables",
    "save_document",
    "save_sections",
    "get_sections_by_doc",
    "list_documents",
    "delete_document",
]
