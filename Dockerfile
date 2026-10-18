# One image for both the FastAPI backend and the Streamlit UI (the compose file
# picks the command). Dependencies install from the locked uv.lock.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv \
    && useradd --create-home app \
    && mkdir -p /app/uploads \
    && chown -R app:app /app

USER app
WORKDIR /app

# Dependencies first, so code edits don't invalidate this (slow) layer.
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000 8501
CMD ["uvicorn", "pdf_chat_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
