"""Vector store (Chroma) + Gemini embeddings for RAG over the wiki.

The wiki (markdown files) is the source of truth. This module is a searchable
*projection* of it: each generated wiki page is chunked, embedded and stored here,
keyed by its ``category/slug`` ref. ``wiki_sync`` keeps the two in step — every
time a page is written or deleted, its chunks are re-embedded or removed. The chat
agent queries :func:`search` to ground its answers.
"""
from __future__ import annotations

import logging

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import settings

logger = logging.getLogger(__name__)

_embeddings: GoogleGenerativeAIEmbeddings | None = None
_client: chromadb.ClientAPI | None = None
_collection = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    return _embeddings


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return _client


def get_collection():
    """Lazily connect to Chroma and return the wiki-knowledge collection."""
    global _collection
    if _collection is None:
        _collection = _get_client().get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_page(ref: str, title: str, category: str, chunks: list[str]) -> int:
    """(Re)index one wiki page: replace any existing chunks for ``ref`` with new ones."""
    remove_page(ref)
    if not chunks:
        return 0
    embeddings = _get_embeddings().embed_documents(chunks)
    ids = [f"{ref}::{i}" for i in range(len(chunks))]
    get_collection().upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {"ref": ref, "title": title, "category": category, "chunk": i}
            for i in range(len(chunks))
        ],
    )
    logger.info("Indexed %d chunk(s) for wiki page '%s'", len(chunks), ref)
    return len(chunks)


def remove_page(ref: str) -> None:
    """Remove all chunks belonging to a single wiki page."""
    try:
        get_collection().delete(where={"ref": ref})
    except Exception:  # pragma: no cover - best effort
        logger.exception("Failed to remove chunks for wiki page '%s'", ref)


def clear() -> None:
    """Drop the entire collection (used when the whole wiki is cleared)."""
    global _collection
    try:
        _get_client().delete_collection(settings.chroma_collection)
    except Exception:  # pragma: no cover - collection may not exist yet
        logger.exception("Failed to clear the vector store collection")
    finally:
        _collection = None


def count() -> int:
    """Number of chunks currently indexed (0 if the store is empty/unreachable)."""
    return get_collection().count()


def search(question: str, k: int | None = None) -> list[dict]:
    """Return the most relevant wiki chunks for a question (above the relevance floor)."""
    k = k or settings.rag_top_k
    embedding = _get_embeddings().embed_query(question)
    res = get_collection().query(
        query_embeddings=[embedding],
        n_results=k,
        include=["metadatas", "documents", "distances"],
    )

    out: list[dict] = []
    metadatas = (res.get("metadatas") or [[]])[0]
    documents = (res.get("documents") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]
    for meta, doc, dist in zip(metadatas, documents, distances):
        similarity = 1.0 - float(dist)  # cosine distance -> similarity
        if similarity < settings.rag_min_relevance:
            continue
        out.append({"document": doc, "metadata": meta, "similarity": similarity})
    return out
