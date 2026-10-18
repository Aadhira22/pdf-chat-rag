import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src import rag
from .custom_types import Hit, RAGQueryResult

# `pdf_chat_rag.main` is also the name of the CLI function exported by the
# package's __init__, which shadows the submodule in a plain `from ... import`.
main = importlib.import_module("pdf_chat_rag.main")

HIT = Hit(text="Paris is the capital of France.", source="geo.pdf", page="3", score=0.8)


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    main.limiter.reset()


@pytest.fixture
def client():
    return TestClient(main.app)


class FakeStore:
    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.deleted = []

    def list_sources(self):
        return [{"source": s, "chunks": n} for s, n in sorted(self.docs.items())]

    def count_source(self, source):
        return self.docs.get(source, 0)

    def delete_source(self, source):
        self.deleted.append(source)
        self.docs.pop(source, None)


# ------------------------------------------------------------------- health
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------- documents
def test_list_documents(client, monkeypatch):
    monkeypatch.setattr(main, "get_store", lambda: FakeStore({"b.pdf": 2, "a.pdf": 5}))
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == {
        "documents": [
            {"source": "a.pdf", "chunks": 5},
            {"source": "b.pdf", "chunks": 2},
        ]
    }


def test_delete_document_removes_chunks_and_file(client, monkeypatch, tmp_path):
    store = FakeStore({"geo.pdf": 4})
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(main.settings, "uploads_dir", str(tmp_path))
    pdf = tmp_path / "geo.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    resp = client.delete("/documents/geo.pdf")

    assert resp.status_code == 200
    assert resp.json() == {"deleted": "geo.pdf", "chunks_removed": 4}
    assert store.deleted == ["geo.pdf"]
    assert not pdf.exists()


def test_delete_unknown_document_is_404(client, monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(main, "get_store", lambda: store)
    assert client.delete("/documents/nope.pdf").status_code == 404
    assert store.deleted == []


def test_delete_rejects_path_traversal(client, monkeypatch):
    store = FakeStore({"x.pdf": 1})
    monkeypatch.setattr(main, "get_store", lambda: store)
    # The HTTP layer already normalizes "/documents/.." away; either way, nothing
    # may be deleted and the request must be refused.
    for path in ["/documents/..", "/documents/%2e%2e%2fsecret.pdf", "/documents/..%5Cx.pdf"]:
        assert client.delete(path).status_code in (400, 404)
    assert store.deleted == []


# -------------------------------------------------------------------- query
def test_query_returns_answer(client, monkeypatch):
    async def fake_answer(question, top_k, source):
        assert (question, top_k, source) == ("capital?", 3, "geo.pdf")
        return RAGQueryResult(answer="Paris [1]", sources=["geo.pdf, p.3"], num_contexts=1)

    monkeypatch.setattr(rag, "answer_question", fake_answer)
    resp = client.post(
        "/query", json={"question": "capital?", "top_k": 3, "source": "geo.pdf"}
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Paris [1]"


def test_query_validates_input(client):
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "hi", "top_k": 99}).status_code == 422


def test_query_stream_sends_tokens_and_sources(client, monkeypatch):
    async def fake_retrieve(question, top_k, source):
        return [HIT]

    async def fake_tokens(question, hits):
        for t in ["Paris", " is", " it [1]."]:
            yield t

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "stream_tokens", fake_tokens)

    resp = client.post("/query/stream", json={"question": "capital?"})

    assert resp.status_code == 200
    assert resp.text == "Paris is it [1]."
    assert json.loads(resp.headers["x-sources"]) == ["geo.pdf, p.3"]
    assert resp.headers["x-num-contexts"] == "1"


def test_query_stream_without_hits_skips_the_llm(client, monkeypatch):
    async def fake_retrieve(question, top_k, source):
        return []

    async def boom(question, hits):
        raise AssertionError("LLM must not be called when nothing was retrieved")
        yield  # pragma: no cover

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "stream_tokens", boom)

    resp = client.post("/query/stream", json={"question": "anything"})

    assert resp.status_code == 200
    assert resp.text == rag.NO_RESULTS_MESSAGE
    assert json.loads(resp.headers["x-sources"]) == []


def test_query_stream_reports_midstream_failure(client, monkeypatch):
    async def fake_retrieve(question, top_k, source):
        return [HIT]

    async def failing_tokens(question, hits):
        yield "Partial"
        raise RuntimeError("openai exploded")

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "stream_tokens", failing_tokens)

    resp = client.post("/query/stream", json={"question": "q"})

    assert resp.status_code == 200
    assert resp.text.startswith("Partial")
    assert "interrupted" in resp.text


def test_query_stream_retrieval_failure_is_502(client, monkeypatch):
    async def failing_retrieve(question, top_k, source):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(rag, "retrieve", failing_retrieve)
    assert client.post("/query/stream", json={"question": "q"}).status_code == 502


# ------------------------------------------------------------- rate limiting
def test_query_is_rate_limited(client, monkeypatch):
    async def fake_answer(question, top_k, source):
        return RAGQueryResult(answer="ok", sources=[], num_contexts=0)

    monkeypatch.setattr(rag, "answer_question", fake_answer)
    statuses = [
        client.post("/query", json={"question": "q"}).status_code for _ in range(11)
    ]
    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429


def test_health_is_not_rate_limited(client):
    assert all(client.get("/health").status_code == 200 for _ in range(30))


# ---------------------------------------------------------------- rag helpers
def test_build_messages_numbers_context_and_labels_sources():
    msgs = rag.build_messages("capital?", [HIT])
    assert msgs[0]["role"] == "system"
    assert "[1] (geo.pdf, p.3)" in msgs[1]["content"]
    assert "Question: capital?" in msgs[1]["content"]


def test_source_label_omits_missing_page():
    assert rag.source_label(Hit(text="t", source="a.pdf", page=None, score=0.5)) == "a.pdf"


def test_stream_tokens_skips_empty_deltas(monkeypatch):
    import asyncio

    def chunk(content, choices=True):
        delta = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if choices else [])

    async def fake_stream():
        for c in [chunk("Hel"), chunk(None), chunk("lo"), chunk("", choices=False)]:
            yield c

    class FakeCompletions:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True
            return fake_stream()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(rag.data_loader, "client", fake_client)

    async def collect():
        return [t async for t in rag.stream_tokens("q", [HIT])]

    assert asyncio.run(collect()) == ["Hel", "lo"]
