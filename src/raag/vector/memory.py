"""In-memory vector store for testing."""

import math
from raag.protocols import VectorItem, VectorSearchResult, VectorStore


class InMemoryVectorStore:
    """In-memory vector store for testing and development.

    Stores vectors in nested dicts: {tenant_id: {id: (vector, metadata)}}
    Provides cosine similarity search with tenant isolation. The first
    upsert fixes the vector dimension for the store.
    """

    def __init__(self) -> None:
        # Dict[tenant_id, Dict[id, (vector, metadata)]]
        self._store: dict[str, dict[str, tuple[list[float], dict]]] = {}
        self._dimension: int | None = None

    async def upsert(self, items: list[VectorItem]) -> None:
        """Insert or update vectors.

        Raises:
            ValueError: If any vector's length differs from the store's
                dimension. Nothing is stored in that case.
        """
        dimension = self._dimension
        for item in items:
            if dimension is None:
                dimension = len(item.vector)
            elif len(item.vector) != dimension:
                raise ValueError(
                    f"Vector dimension mismatch: vector '{item.id}' has {len(item.vector)} "
                    f"dimensions, store has {dimension}"
                )
        self._dimension = dimension
        for item in items:
            if item.tenant_id not in self._store:
                self._store[item.tenant_id] = {}
            self._store[item.tenant_id][item.id] = (item.vector, item.metadata)

    async def search(
        self,
        vector: list[float],
        k: int,
        tenant_id: str
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using cosine similarity."""
        if tenant_id not in self._store:
            return []

        results = []
        for id, (vec, meta) in self._store[tenant_id].items():
            score = self._cosine_similarity(vector, vec)
            results.append(VectorSearchResult(id=id, score=score, metadata=meta))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    async def delete(self, ids: list[str], tenant_id: str) -> None:
        """Delete vectors by ID."""
        if tenant_id not in self._store:
            return
        for id in ids:
            self._store[tenant_id].pop(id, None)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            raise ValueError(
                f"Vector dimension mismatch: query has {len(a)} dimensions, stored vector has {len(b)}"
            )
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
