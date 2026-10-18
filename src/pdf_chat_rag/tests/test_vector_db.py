import pytest

from src import vector_db


class FakeQdrantClient:
    """Minimal stand-in: one collection whose vector size we control."""

    size = 1536
    exists = True
    points: list = []

    def __init__(self, url, timeout):
        self.created = None

    def collection_exists(self, name):
        return self.exists

    def create_collection(self, collection_name, vectors_config):
        self.created = vectors_config

    def get_collection(self, name):
        vectors = type("V", (), {"size": self.size})()
        params = type("P", (), {"vectors": vectors})()
        return type("I", (), {"config": type("C", (), {"params": params})()})()

    def scroll(self, collection, limit, offset, with_payload, with_vectors):
        # two pages, to exercise the pagination loop
        if offset is None:
            return self.points[:2], "next"
        return self.points[2:], None


def _point(source):
    return type("Pt", (), {"payload": {"source": source}})()


def test_creates_collection_with_configured_dimension(monkeypatch):
    FakeQdrantClient.exists = False
    monkeypatch.setattr(vector_db, "QdrantClient", FakeQdrantClient)
    store = vector_db.QdrantStorage()
    assert store.client.created.size == vector_db.settings.embed_dim == 1536


def test_dimension_mismatch_raises_helpful_error(monkeypatch):
    FakeQdrantClient.exists = True
    FakeQdrantClient.size = 768  # e.g. a leftover nomic-embed-text collection
    monkeypatch.setattr(vector_db, "QdrantClient", FakeQdrantClient)
    with pytest.raises(RuntimeError, match="re-ingest"):
        vector_db.QdrantStorage()


def test_list_sources_counts_across_pages(monkeypatch):
    FakeQdrantClient.exists = True
    FakeQdrantClient.size = 1536
    FakeQdrantClient.points = [_point("b.pdf"), _point("a.pdf"), _point("b.pdf"), _point("b.pdf")]
    monkeypatch.setattr(vector_db, "QdrantClient", FakeQdrantClient)
    assert vector_db.QdrantStorage().list_sources() == [
        {"source": "a.pdf", "chunks": 1},
        {"source": "b.pdf", "chunks": 3},
    ]
