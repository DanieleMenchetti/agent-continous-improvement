"""Keeps the vector store in sync with the wiki (the source of truth).

Whenever a wiki page is written, edited or deleted, the corresponding call here
re-embeds or removes it in the vector store, so RAG always reflects the current
wiki. All operations are best-effort: a vector-store hiccup must never break a
wiki write (the markdown on disk remains authoritative and can be re-synced).

Only the *generated* pages are indexed — ``summaries``, ``entities``, ``concepts``
and ``feedback``. The raw ``sources/`` dumps are excluded: they are large, immutable,
and already distilled into the generated pages.
"""
from __future__ import annotations

import logging

from . import vectorstore, wiki

logger = logging.getLogger(__name__)

# Categories whose pages are embedded for retrieval (excludes raw sources).
INDEXED_CATEGORIES = wiki.CATEGORIES

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _chunk(text: str) -> list[str]:
    """Split page text into overlapping chunks of ~CHUNK_SIZE chars."""
    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def _page_chunks(title: str, body: str) -> list[str]:
    # Prefix the title so every chunk carries the topic it belongs to.
    return _chunk(f"{title}\n\n{body.strip()}")


def index_page(category: str, slug: str) -> None:
    """(Re)embed a single wiki page. No-op for non-indexed categories."""
    if category not in INDEXED_CATEGORIES:
        return
    page = wiki.read_page(category, slug)
    if page is None:
        return
    meta, body = page
    title = str(meta.get("title") or slug)
    ref = f"{category}/{slug}"
    try:
        vectorstore.upsert_page(ref, title, category, _page_chunks(title, body))
    except Exception:  # never let a sync failure break the wiki write
        logger.exception("Failed to sync wiki page '%s' to the vector store", ref)


def index_refs(refs: list[str]) -> None:
    """Index a batch of ``category/slug`` refs (e.g. all pages an ingest touched)."""
    for ref in refs:
        if "/" in ref:
            category, slug = ref.split("/", 1)
            index_page(category, slug)


def remove_page(category: str, slug: str) -> None:
    try:
        vectorstore.remove_page(f"{category}/{slug}")
    except Exception:
        logger.exception("Failed to remove wiki page '%s/%s' from the vector store", category, slug)


def clear() -> None:
    try:
        vectorstore.clear()
    except Exception:
        logger.exception("Failed to clear the vector store")


def reindex_all() -> int:
    """Rebuild the whole vector store from the wiki. Returns the page count indexed."""
    clear()
    n = 0
    for category in INDEXED_CATEGORIES:
        for slug in wiki.list_pages(category):
            index_page(category, slug)
            n += 1
    logger.info("Reindexed %d wiki page(s) into the vector store", n)
    return n


def ensure_indexed() -> None:
    """Startup reconcile: if the store is empty but the wiki has pages, build it.

    Best-effort so the app still boots if Chroma or the embedding API is briefly
    unavailable — a later wiki edit (or a manual reindex) will resync.
    """
    try:
        if vectorstore.count() == 0 and any(wiki.list_pages(c) for c in INDEXED_CATEGORIES):
            logger.info("Vector store is empty; building it from the existing wiki…")
            reindex_all()
    except Exception:
        logger.exception("Startup vector-store reconcile failed; skipping")
