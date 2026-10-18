from collections import Counter
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .config import settings
from .custom_types import Hit


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


class QdrantStorage:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        self.collection = settings.qdrant_collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embed_dim, distance=Distance.COSINE
                ),
            )
        else:
            self._check_dimension()

    def _check_dimension(self) -> None:
        """Fail loudly if the collection was built with a different embedding model."""
        vectors = self.client.get_collection(self.collection).config.params.vectors
        size = getattr(vectors, "size", None)  # None for named-vector collections
        if size is not None and size != settings.embed_dim:
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' has {size}-dim vectors but "
                f"EMBED_DIM={settings.embed_dim}. Switching embedding models needs a "
                "fresh collection: delete it (or set QDRANT_COLLECTION to a new name) "
                "and re-ingest your documents."
            )

    def delete_source(self, source: str) -> None:
        """Removes all chunks of one document (also used before re-ingesting it)."""
        self.client.delete(
            self.collection,
            points_selector=FilterSelector(filter=_source_filter(source)),
        )

    def count_source(self, source: str) -> int:
        return self.client.count(
            self.collection, count_filter=_source_filter(source), exact=True
        ).count

    def list_sources(self) -> list[dict]:
        """Distinct document names with their chunk counts, sorted by name."""
        counts: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for p in points:
                if p.payload and p.payload.get("source"):
                    counts[p.payload["source"]] += 1
            if offset is None:
                break
        return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(self.collection, points=points)

    def search(
        self, query_vector, top_k: int = 5, source: str | None = None
    ) -> list[Hit]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=_source_filter(source) if source else None,
            score_threshold=settings.min_score,
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )
        return [
            Hit(
                text=p.payload.get("text", ""),
                source=p.payload.get("source", ""),
                page=p.payload.get("page"),
                score=p.score,
            )
            for p in response.points
            if p.payload and p.payload.get("text")
        ]


@lru_cache(maxsize=1)
def get_store() -> QdrantStorage:
    """One shared client instead of a new connection on every call."""
    return QdrantStorage()
