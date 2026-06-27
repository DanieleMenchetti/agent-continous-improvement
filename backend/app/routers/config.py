"""Read-only endpoint that exposes non-sensitive runtime settings."""
from fastapi import APIRouter

from ..config import settings

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("")
def get_config() -> dict:
    return {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "chroma_collection": settings.chroma_collection,
        "rag_top_k": settings.rag_top_k,
        "rag_min_relevance": settings.rag_min_relevance,
    }
