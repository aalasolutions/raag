"""Qdrant vector store adapter implementing VectorStore protocol."""

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    FilterSelector,
    HasIdCondition,
    MatchValue,
)
from raag.protocols import VectorItem, VectorSearchResult


class QdrantVectorStore:
    """Qdrant-backed vector store with tenant isolation.

    Uses payload filtering for tenant isolation rather than
    separate collections per tenant. Store operations use
    AsyncQdrantClient so they do not block the event loop.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "raag",
        vector_size: int = 1024,
    ):
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._ensure_collection(url)
        self._client = AsyncQdrantClient(url=url)

    def _ensure_collection(self, url: str):
        client = QdrantClient(url=url)
        try:
            collections = [c.name for c in client.get_collections().collections]
            if self._collection_name not in collections:
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
        finally:
            client.close()

    async def upsert(self, items: list[VectorItem]) -> None:
        if not items:
            return
        points = [
            PointStruct(
                id=item.id,
                vector=item.vector,
                payload={
                    "tenant_id": item.tenant_id,
                    **item.metadata,
                },
            )
            for item in items
        ]
        await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def search(
        self,
        vector: list[float],
        k: int,
        tenant_id: str,
    ) -> list[VectorSearchResult]:
        results = await self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    )
                ]
            ),
            limit=k,
        )
        return [
            VectorSearchResult(
                id=point.id,
                score=point.score,
                metadata={k: v for k, v in point.payload.items() if k != "tenant_id"},
            )
            for point in results.points
        ]

    async def delete(self, ids: list[str], tenant_id: str) -> None:
        if not ids:
            return
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        HasIdCondition(has_id=ids),
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id),
                        ),
                    ]
                )
            ),
        )
