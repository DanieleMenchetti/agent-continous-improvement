"""Config endpoints: Agent Soul prompt and document ingestion."""
import io
import re

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from .. import ingest_agent, lint_agent, wiki, wiki_sync
from ..database import get_db
from ..models import KeyValueSetting
from ..schemas import LintReport, SoulPrompt, WikiPageEdit

router = APIRouter(prefix="/api/config", tags=["config"])

SOUL_KEY = "agent_soul"


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

    # Incrementally build/extend the markdown wiki from this document. This does
    # blocking network I/O (many LLM calls), so run it off the event loop.
    wiki_result = await run_in_threadpool(
        ingest_agent.ingest_document, file.filename, full_text
    )

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
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
    wiki_sync.index_page(category, safe_slug)  # keep RAG in sync with the edit
    return (wiki._root() / category / f"{safe_slug}.md").read_text(encoding="utf-8")


@router.delete("/wiki/{category}/{slug}")
def delete_wiki_page(category: str, slug: str) -> dict:
    """Delete a single wiki page."""
    safe_slug = _resolve_target(category, slug)
    if not wiki.page_exists(category, safe_slug):
        raise HTTPException(status_code=404, detail="Wiki page not found.")

    wiki.delete_page(category, safe_slug)
    wiki.regenerate_index()
    wiki_sync.remove_page(category, safe_slug)  # drop it from RAG too
    return {"ok": True, "stats": wiki.stats()}


@router.delete("/wiki")
def clear_wiki() -> dict:
    """Delete the entire knowledge base: all pages, sources, index and log,
    and clear the vector store that mirrors it."""
    wiki.clear_all()
    wiki_sync.clear()
    return {"ok": True, "stats": wiki.stats()}


@router.post("/wiki/reindex")
async def reindex_wiki() -> dict:
    """Rebuild the vector store from scratch to match the current wiki.

    Normally the store stays in sync automatically on every wiki change; this is a
    manual reconcile for recovery. It re-embeds every page, so it runs off the event
    loop.
    """
    pages = await run_in_threadpool(wiki_sync.reindex_all)
    return {"ok": True, "pages_indexed": pages}
