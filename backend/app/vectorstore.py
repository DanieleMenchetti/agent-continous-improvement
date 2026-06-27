"""Vector store for expert feedback (Chroma) + Gemini embeddings.

Each feedback entry is stored as one document whose *embedding* is computed from
the original customer question (so semantically similar future questions match),
while the stored document text is the expert's guidance used as RAG context.
"""
from __future__ import annotations

import logging

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .config import settings

logger = logging.getLogger(__name__)

_embeddings: GoogleGenerativeAIEmbeddings | None = None
_collection = None


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    return _embeddings


def get_collection():
    """Lazily connect to Chroma and return the feedback collection."""
    global _collection
    if _collection is None:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _doc_id(feedback_id: int) -> str:
    return f"feedback-{feedback_id}"


def upsert_feedback(
    *,
    feedback_id: int,
    question: str,
    answer: str,
    feedback_content: str,
    is_correction: bool,
    rating: int | None,
) -> None:
    """Insert or update a feedback document in the vector store."""
    embedding = _get_embeddings().embed_query(question)
    document = (
        f"When a customer asks something like: \"{question}\"\n"
        f"Expert guidance: {feedback_content}"
    )
    get_collection().upsert(
        ids=[_doc_id(feedback_id)],
        embeddings=[embedding],
        documents=[document],
        metadatas=[
            {
                "feedback_id": feedback_id,
                "question": question,
                "original_answer": answer,
                "feedback": feedback_content,
                "is_correction": is_correction,
                "rating": rating if rating is not None else -1,
            }
        ],
    )
    logger.info("Upserted feedback %s into vector store", feedback_id)


def upsert_document_chunks(filename: str, chunks: list[str]) -> int:
    """Embed and store text chunks extracted from an uploaded document."""
    embeddings = _get_embeddings().embed_documents(chunks)
    ids = [f"doc-{filename}-{i}" for i in range(len(chunks))]
    get_collection().upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": filename, "chunk": i, "type": "document"} for i in range(len(chunks))],
    )
    logger.info("Upserted %d chunks from document '%s'", len(chunks), filename)
    return len(chunks)


def delete_feedback(feedback_id: int) -> None:
    try:
        get_collection().delete(ids=[_doc_id(feedback_id)])
    except Exception:  # pragma: no cover - best effort cleanup
        logger.exception("Failed to delete feedback %s from vector store", feedback_id)


def search(question: str, k: int | None = None) -> list[dict]:
    """Return the most relevant expert feedback for a customer question."""
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
        # cosine distance -> similarity
        similarity = 1.0 - float(dist)
        if similarity < settings.rag_min_relevance:
            continue
        out.append({"document": doc, "metadata": meta, "similarity": similarity})
    return out
