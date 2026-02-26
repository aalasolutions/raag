"""VectorStore Protocol for pluggable vector backends."""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class VectorItem:
    """A single vector with metadata for storage."""
    id: str
    vector: list[float]
    metadata: dict
    tenant_id: str


@dataclass
class VectorSearchResult:
    """Result from a vector search."""
    id: str
    score: float
    metadata: dict


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.

    Consumers can implement their own vector store (Qdrant, Weaviate, etc.)
    by implementing this interface.
    """

    async def upsert(self, items: list[VectorItem]) -> None:
        """Insert or update vectors.

        Args:
            items: List of vectors to store
        """
        ...

    async def search(
        self,
        vector: list[float],
        k: int,
        tenant_id: str
    ) -> list[VectorSearchResult]:
        """Search for similar vectors.

        Args:
            vector: Query vector
            k: Number of results to return
            tenant_id: Tenant to search within

        Returns:
            List of search results sorted by score (descending)
        """
        ...

    async def delete(self, ids: list[str], tenant_id: str) -> None:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete
            tenant_id: Tenant to delete from
        """
        ...
