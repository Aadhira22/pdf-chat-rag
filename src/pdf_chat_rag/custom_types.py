from typing import Literal

import pydantic

class Chunk(pydantic.BaseModel):
    text: str
    page: str | None = None


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[Chunk]
    source_id: str | None = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class Hit(pydantic.BaseModel):
    text: str
    source: str
    page: str | None = None
    score: float


class RAGSearchResult(pydantic.BaseModel):
    hits: list[Hit]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int

class ChatTurn(pydantic.BaseModel):
    """One earlier message in the conversation, sent by the client with each query."""

    role: Literal["user", "assistant"]
    content: str = pydantic.Field(max_length=4000)