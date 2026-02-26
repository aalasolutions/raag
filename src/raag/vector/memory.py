"""In-memory vector store for testing."""

import math
from raag.protocols import VectorItem, VectorSearchResult, VectorStore


class InMemoryVectorStore:
    """In-memory vector store for testing and development.

    Stores vectors in nested dicts: {tenant_id: {id: (vector, metadata)}}
    Provides cosine similarity search with tenant isolation.
    """

    def __init__(self) -> None:
        # Dict[tenant_id, Dict[id, (vector, metadata)]]
        self._store: dict[str, dict[str, tuple[list[float], dict]]] = {}

    async def upsert(self, items: list[VectorItem]) -> None:
        """Insert or update vectors."""
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
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
