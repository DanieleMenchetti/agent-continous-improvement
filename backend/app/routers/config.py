"""Config endpoints: Agent Soul prompt and document ingestion."""
import io
import re

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from .. import ingest_agent, lint_agent, vectorstore, wiki
from ..database import get_db
from ..models import KeyValueSetting
from ..schemas import LintReport, SoulPrompt, WikiPageEdit

router = APIRouter(prefix="/api/config", tags=["config"])

SOUL_KEY = "agent_soul"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Agent Soul ────────────────────────────────────────────────────────────────

@router.get("/soul", response_model=SoulPrompt)
def get_soul(db: Session = Depends(get_db)) -> SoulPrompt:
    row = db.get(KeyValueSetting, SOUL_KEY)
    return SoulPrompt(prompt=row.value if row else "")


@router.put("/soul", response_model=SoulPrompt)
def put_soul(body: SoulPrompt, db: Session = Depends(get_db)) -> SoulPrompt:
    row = db.get(KeyValueSetting, SOUL_KEY)
    if row is None:
        row = KeyValueSetting(key=SOUL_KEY, value=body.prompt)
        db.add(row)
    else:
        row.value = body.prompt
    db.commit()
    return SoulPrompt(prompt=row.value)


# ── Document ingestion ────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of ~CHUNK_SIZE chars."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


@router.post("/documents")
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf not installed.")

    raw = await file.read()
    reader = PdfReader(io.BytesIO(raw))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    chunks = _chunk_text(full_text)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename)
    # These do blocking network I/O (embeddings + many LLM calls); run them off the
    # event loop so the server stays responsive during a long ingest.
    count = await run_in_threadpool(vectorstore.upsert_document_chunks, safe_name, chunks)

    # Incrementally build/extend the markdown wiki from this document.
    wiki_result = await run_in_threadpool(
        ingest_agent.ingest_document, file.filename, full_text
    )

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks_indexed": count,
        "wiki": wiki_result,
    }


# ── Wiki browsing (read-only) ───────────────────────────────────────────────

@router.get("/wiki")
def list_wiki() -> dict:
    """List the wiki page catalog (sources + generated pages) and stats.

    Each page entry carries ``{slug, title, summary, updated}`` read from its
    frontmatter, so a client can render a table of contents without fetching
    every page individually.
    """
    wiki.ensure_wiki()
    return {
        "stats": wiki.stats(),
        "sources": wiki.list_page_infos(wiki.SOURCES_DIR),
        **{cat: wiki.list_page_infos(cat) for cat in wiki.CATEGORIES},
    }


@router.post("/wiki/lint", response_model=LintReport)
async def lint_wiki() -> LintReport:
    """Run a health check over the wiki and return a report of any findings.

    Mirrors the periodic "lint" from karpathy's LLM-wiki guidelines: structural
    checks (schema integrity, broken links, orphan pages) plus an LLM pass for
    contradictions, stale claims, coverage gaps, missing cross-references, and
    data gaps. The LLM pass is blocking network I/O, so it runs off the event loop.
    """
    report = await run_in_threadpool(lint_agent.lint_wiki)
    return report


def _resolve_target(category: str, slug: str) -> str:
    """Validate a (category, slug) pair and return the sanitized slug.

    Raises 404/400 for an unknown category or a slug that sanitizes to nothing.
    """
    allowed = (*wiki.CATEGORIES, wiki.SOURCES_DIR)
    if category not in allowed:
        raise HTTPException(status_code=404, detail="Unknown wiki category.")
    safe_slug = re.sub(r"[^a-zA-Z0-9._-]", "", slug)
    if not safe_slug:
        raise HTTPException(status_code=400, detail="Invalid page slug.")
    return safe_slug


@router.get("/wiki/{category}/{slug}", response_class=PlainTextResponse)
def get_wiki_page(category: str, slug: str) -> str:
    """Return the raw markdown of a single wiki page."""
    safe_slug = _resolve_target(category, slug)
    if not wiki.page_exists(category, safe_slug):
        raise HTTPException(status_code=404, detail="Wiki page not found.")
    return (wiki._root() / category / f"{safe_slug}.md").read_text(encoding="utf-8")


# ── Knowledge editing / deletion ─────────────────────────────────────────────

@router.put("/wiki/{category}/{slug}", response_class=PlainTextResponse)
def update_wiki_page(category: str, slug: str, body: WikiPageEdit) -> str:
    """Overwrite a page's raw markdown and refresh the index."""
    safe_slug = _resolve_target(category, slug)
    if not wiki.page_exists(category, safe_slug):
        raise HTTPException(status_code=404, detail="Wiki page not found.")
    wiki.write_raw(category, safe_slug, body.content)
    wiki.regenerate_index()
    return (wiki._root() / category / f"{safe_slug}.md").read_text(encoding="utf-8")


@router.delete("/wiki/{category}/{slug}")
def delete_wiki_page(category: str, slug: str) -> dict:
    """Delete a single page. Deleting a source also purges its indexed chunks."""
    safe_slug = _resolve_target(category, slug)
    if not wiki.page_exists(category, safe_slug):
        raise HTTPException(status_code=404, detail="Wiki page not found.")

    # A source owns chunks in the vector store (keyed by original filename).
    if category == wiki.SOURCES_DIR:
        filename = wiki.source_filename(safe_slug)
        if filename:
            vectorstore.delete_document_chunks(filename)

    wiki.delete_page(category, safe_slug)
    wiki.regenerate_index()
    return {"ok": True, "stats": wiki.stats()}


@router.delete("/wiki")
def clear_wiki() -> dict:
    """Delete the entire knowledge base: all pages, sources, index and log,
    plus every ingested-document chunk in the vector store."""
    wiki.clear_all()
    vectorstore.delete_all_document_chunks()
    return {"ok": True, "stats": wiki.stats()}
