from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- OpenAI ---------------------------------------------------------
    # Never commit a real key. Put it in .env (git-ignored) as OPENAI_API_KEY.
    openai_api_key: str = ""
    # Override to point at any OpenAI-compatible endpoint (Azure, Ollama, ...).
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536  # text-embedding-3-small; must match the model

    # --- Qdrant ---------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"

    # --- Retrieval / chunking ------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 100
    embed_batch_size: int = 32
    # OpenAI cosine scores run lower than nomic's. Re-tune with eval/run_eval.py.
    min_score: float = 0.2

    
    # --- Conversation memory -------------------------------------------
    # How many of the most recent chat messages (user + assistant) are used.
    max_history_messages: int = 6
    
    # --- Inngest --------------------------------------------------------
    # Origin of the Inngest dev server. The API must reach it to register its
    # functions, so inside Docker this is http://inngest:8288 (not localhost).
    inngest_url: str = "http://127.0.0.1:8288"

    # --- Files ----------------------------------------------------------
    # Streamlit writes uploads here and the API reads (and deletes) them, so in
    # Docker both containers mount the same volume at this path.
    uploads_dir: str = "uploads"

    # --- Rate limiting (slowapi syntax, per client IP) -------------------
    rate_limit_query: str = "10/minute"
    rate_limit_documents: str = "60/minute"


settings = Settings()
