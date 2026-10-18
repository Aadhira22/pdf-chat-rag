import asyncio
import datetime
import json
import logging
import uuid
from pathlib import Path

import inngest
import inngest.fast_api
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from inngest.experimental import ai
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import rag
from .config import settings
from .custom_types import (
    ChatTurn,
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from .data_loader import embed_texts, load_and_chunk_pdf
from .vector_db import get_store

logger = logging.getLogger("uvicorn")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logger,
    is_production=False,
    api_base_url=settings.inngest_url,
    event_api_base_url=settings.inngest_url,
    serializer=inngest.PydanticSerializer(),
)


# ------------------------------------------------------------------ Inngest
# Ingestion stays a durable workflow. There is deliberately no per-document
# rate limit here: it used to block re-uploading a document within 4 hours (and
# so delete + re-upload). The short throttle remains as a basic anti-spam guard.
@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=1,
        period=datetime.timedelta(seconds=5),
        burst=2,
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    async def _load() -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = await asyncio.to_thread(load_and_chunk_pdf, pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    async def _upsert(data: RAGChunkAndSrc) -> RAGUpsertResult:
        vecs = await embed_texts([c.text for c in data.chunks])
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{data.source_id}:{i}"))
            for i in range(len(data.chunks))
        ]
        payloads = [
            {"source": data.source_id, "page": c.page, "text": c.text}
            for c in data.chunks
        ]
        store = get_store()
        # Sync Qdrant calls: keep them off the event loop.
        await asyncio.to_thread(store.delete_source, data.source_id)
        await asyncio.to_thread(store.upsert, ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(data.chunks))

    data = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    result = await ctx.step.run(
        "embed-and-upsert", lambda: _upsert(data), output_type=RAGUpsertResult
    )
    return result.model_dump()


# Durable, non-streaming query workflow. The Streamlit UI now uses the streaming
# endpoint instead, but this stays registered for observability and event-driven use.
@inngest_client.create_function(
    fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    source = ctx.event.data.get("source")  # optional: restrict to one document

    async def _search() -> RAGSearchResult:
        return RAGSearchResult(hits=await rag.retrieve(question, top_k, source))

    found = await ctx.step.run("embed-and-search", _search, output_type=RAGSearchResult)

    if not found.hits:
        return RAGQueryResult(
            answer=rag.NO_RESULTS_MESSAGE, sources=[], num_contexts=0
        ).model_dump()

    adapter = ai.openai.Adapter(
        auth_key=settings.openai_api_key,
        model=settings.llm_model,
        base_url=settings.openai_base_url,
    )
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": rag.build_messages(question, found.hits),
        },
    )
    return RAGQueryResult(
        answer=res["choices"][0]["message"]["content"].strip(),
        sources=rag.source_labels(found.hits),
        num_contexts=len(found.hits),
    ).model_dump()


# ---------------------------------------------------------------------- API
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="RAG Production App")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    source: str | None = None  # optional: restrict to one document
    # Earlier turns, oldest first. The client owns the conversation; the server
    # stays stateless and only uses the last few messages (MAX_HISTORY_MESSAGES).
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=RAGQueryResult)
@limiter.limit(settings.rate_limit_query)
async def query(request: Request, body: QueryRequest):
    """Non-streaming answer: one JSON response. Handy for API clients."""
    try:
        return await rag.answer_question(
            body.question, body.top_k, body.source, body.history
        )
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=502, detail="Failed to generate an answer.")


@app.post("/query/stream")
@limiter.limit(settings.rate_limit_query)
async def query_stream(request: Request, body: QueryRequest):
    """Streams the answer as plain text as it is generated.

    Retrieval runs first, so the sources are already known and are sent up front
    in the `X-Sources` response header (a JSON array of strings).
    """
    try:
        # A follow-up like "and the second one?" is rewritten into a standalone
        # question first, so the vector search has something meaningful to match.
        search_question = await rag.condense_question(body.question, body.history)
        hits = await rag.retrieve(search_question, body.top_k, body.source)
    except Exception:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail="Failed to search the documents.")

    async def body_stream():
        if not hits:
            yield rag.NO_RESULTS_MESSAGE
            return
        try:
            async for token in rag.stream_tokens(body.question, hits, body.history):
                yield token
        except Exception:
            logger.exception("Streaming failed")
            yield "\n\n[The answer was interrupted by an error.]"

    return StreamingResponse(
        body_stream(),
        media_type="text/plain; charset=utf-8",
        headers={
            # ensure_ascii keeps non-ASCII file names safe inside an HTTP header
            "X-Sources": json.dumps(rag.source_labels(hits)),
            "X-Num-Contexts": str(len(hits)),
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/documents")
@limiter.limit(settings.rate_limit_documents)
async def list_documents(request: Request):
    docs = await asyncio.to_thread(get_store().list_sources)
    return {"documents": docs}


@app.delete("/documents/{source}")
@limiter.limit(settings.rate_limit_documents)
async def delete_document(request: Request, source: str):
    """Remove a document's chunks from Qdrant and its PDF from the uploads dir."""
    if Path(source).name != source or source in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid document name.")

    store = get_store()
    chunks = await asyncio.to_thread(store.count_source, source)
    if chunks == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{source}'.")

    await asyncio.to_thread(store.delete_source, source)
    (Path(settings.uploads_dir) / source).unlink(missing_ok=True)
    return {"deleted": source, "chunks_removed": chunks}


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])