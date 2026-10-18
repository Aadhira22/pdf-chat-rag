"""Retrieval + generation shared by the REST endpoints and the Inngest workflow."""

import asyncio
from collections.abc import AsyncIterator

from . import data_loader
from .config import settings
from .custom_types import ChatTurn, Hit, RAGQueryResult
from .vector_db import get_store

NO_RESULTS_MESSAGE = "I couldn't find anything relevant in the uploaded documents."

SYSTEM_PROMPT = (
    "Answer using only the numbered context. Cite the passages you use like [1]. "
    "If the context does not contain the answer, say you don't know."
)


def source_label(hit: Hit) -> str:
    return f"{hit.source}, p.{hit.page}" if hit.page else hit.source


def source_labels(hits: list[Hit]) -> list[str]:
    return sorted({source_label(h) for h in hits})


CONDENSE_PROMPT = (
    "Rewrite the user's last question as a standalone question that can be "
    "understood without the chat history. Resolve pronouns and references like "
    "'it' or 'the second one'. Do NOT answer it. Return only the question."
)


def recent(history: list[ChatTurn] | None) -> list[ChatTurn]:
    """Only the last few messages: keeps prompts small and costs predictable."""
    return (history or [])[-settings.max_history_messages :]


async def condense_question(question: str, history: list[ChatTurn] | None) -> str:
    """Turn a follow-up into a standalone question so the vector search can use it.

    "What about its second chapter?" means nothing to Qdrant on its own. With no
    history this is a no-op and costs nothing. If the rewrite fails for any
    reason we fall back to the original question instead of failing the request.
    """
    turns = recent(history)
    if not turns:
        return question
    transcript = "\n".join(f"{t.role}: {t.content}" for t in turns)
    try:
        resp = await data_loader.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": CONDENSE_PROMPT},
                {
                    "role": "user",
                    "content": f"Chat history:\n{transcript}\n\nLast question: {question}",
                },
            ],
            temperature=0,
            max_tokens=150,
        )
        return (resp.choices[0].message.content or "").strip() or question
    except Exception:
        return question


def build_messages(
    question: str, hits: list[Hit], history: list[ChatTurn] | None = None
) -> list[dict]:
    context_block = "\n\n".join(
        f"[{i}] ({source_label(h)})\n{h.text}" for i, h in enumerate(hits, 1)
    )
    past = [{"role": t.role, "content": t.content} for t in recent(history)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *past,
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        },
    ]


async def retrieve(question: str, top_k: int, source: str | None) -> list[Hit]:
    qvec = (await data_loader.embed_texts([question]))[0]
    # The Qdrant client is synchronous; keep it off the event loop.
    return await asyncio.to_thread(get_store().search, qvec, top_k, source)


async def answer_question(
    question: str,
    top_k: int = 5,
    source: str | None = None,
    history: list[ChatTurn] | None = None,
) -> RAGQueryResult:
    """Non-streaming answer (POST /query)."""
    hits = await retrieve(await condense_question(question, history), top_k, source)
    if not hits:
        return RAGQueryResult(answer=NO_RESULTS_MESSAGE, sources=[], num_contexts=0)

    resp = await data_loader.client.chat.completions.create(
        model=settings.llm_model,
        messages=build_messages(question, hits, history),
        temperature=0.2,
        max_tokens=1024,
    )
    return RAGQueryResult(
        answer=(resp.choices[0].message.content or "").strip(),
        sources=source_labels(hits),
        num_contexts=len(hits),
    )


async def stream_tokens(
    question: str, hits: list[Hit], history: list[ChatTurn] | None = None
) -> AsyncIterator[str]:
    """Yield the answer piece by piece as the LLM produces it."""
    stream = await data_loader.client.chat.completions.create(
        model=settings.llm_model,
        messages=build_messages(question, hits, history),
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content