from openai import AsyncOpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from .config import settings
from .custom_types import Chunk

# A placeholder is used when no key is set so that importing this module (e.g.
# in unit tests) never fails; real calls will get a clear 401 from OpenAI.
client = AsyncOpenAI(
    base_url=settings.openai_base_url,
    api_key=settings.openai_api_key or "missing-openai-api-key",
)
splitter = SentenceSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


def load_and_chunk_pdf(path: str) -> list[Chunk]:
    """Sync and CPU-bound. Call it with asyncio.to_thread."""
    chunks: list[Chunk] = []
    for doc in PDFReader().load_data(file=path):
        if not doc.text:
            continue
        page = doc.metadata.get("page_label")
        for text in splitter.split_text(doc.text):
            chunks.append(Chunk(text=text, page=page))

    return chunks


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches. OpenAI embedding models need no task prefixes."""
    vectors: list[list[float]] = []
    size = settings.embed_batch_size
    for i in range(0, len(texts), size):
        batch = texts[i : i + size]
        resp = await client.embeddings.create(model=settings.embed_model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    return vectors
