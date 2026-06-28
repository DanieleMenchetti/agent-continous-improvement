"""Application configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Google / Gemini
    google_api_key: str = ""
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"

    # PostgreSQL (relational store for Q&A pairs + feedback metadata)
    database_url: str = "postgresql+psycopg2://app:app@postgres:5432/app"

    # Chroma (vector store for expert feedback used in RAG)
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    chroma_collection: str = "expert_feedback"

    # Retrieval
    rag_top_k: int = 4
    rag_min_relevance: float = 0.0  # 0..1 similarity floor (0 = keep all)

    # Wiki (LLM-maintained markdown knowledge base built from ingested documents)
    wiki_dir: str = "/data/wiki"


settings = Settings()
