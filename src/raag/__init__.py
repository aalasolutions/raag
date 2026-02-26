"""RAAG: Relationship-Aware Augmented Generation - A minimal multi-tenant RAG library."""

from raag.core import Raag
from raag.config import RaagConfig
from raag.tenant.context import tenant_context, require_tenant, get_current_tenant
from raag.models.document import Document
from raag.models.section import Section
from raag.models.results import SearchResult
from raag.protocols import VectorStore, VectorItem, VectorSearchResult
from raag.vector.memory import InMemoryVectorStore
from raag.splitters.base import Splitter, SplitSection
from raag.splitters.markdown import MarkdownSplitter

__version__ = "0.1.3"

__all__ = [
    # Core
    "Raag",
    "RaagConfig",
    # Tenant context
    "tenant_context",
    "require_tenant",
    "get_current_tenant",
    # Models
    "Document",
    "Section",
    "SearchResult",
    # Protocols
    "VectorStore",
    "VectorItem",
    "VectorSearchResult",
    # Implementations
    "InMemoryVectorStore",
    # Splitters
    "Splitter",
    "SplitSection",
    "MarkdownSplitter",
]
