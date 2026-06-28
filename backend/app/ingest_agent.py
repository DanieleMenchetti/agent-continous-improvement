"""Fixed-pipeline ingestion agent that turns a source document into wiki pages.

Pipeline (per ingested document):
    1. summarize    — produce a title, one-line summary, and a thorough overview
    2. extract      — IDENTIFY entities and a SMALL number of broad concepts (just
                      title/kind/summary + a coverage checklist — NOT the full prose)
    3. compose      — for EACH item, a dedicated pass reads the full source and
                      writes a comprehensive page retaining >=90% of the relevant info
    4. write/merge  — create new pages, or *non-destructively merge* into existing
                      ones (prior content is preserved; new info is integrated)
    5. reindex      — regenerate index.md
    6. log          — append an entry to log.md

Design goals:
  * No information loss — the ENTIRE document is processed via overlapping windows
    (never silently truncated), and each page is COMPOSED in its own dedicated pass
    against the full source so it can retain >=90% of the relevant material rather
    than a few sentences.
  * Few, detailed concept pages — extraction consolidates related sub-topics into
    broad pages rather than producing many thin ones; the compose pass then makes
    each one exhaustive.
  * Incremental & non-destructive — sources are immutable, the log is append-only,
    and existing pages are merged, never replaced.

Why a separate compose pass: when one structured-output call must emit every item's
full prose at once, the model spreads a limited output budget across all items and
compresses each into a few sentences. Identifying items cheaply, then writing each
page in its own call, gives every page the model's full output budget.
"""
from __future__ import annotations

import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from . import wiki
from .config import settings

logger = logging.getLogger(__name__)

# The whole document is processed: long documents are split into overlapping
# windows so nothing is dropped. Windows are large (Gemini handles big context),
# so most documents are a single window.
WINDOW_CHARS = 60_000
WINDOW_OVERLAP = 2_000

# The compose pass writes each page against the FULL source so it can retain
# everything relevant. Gemini's context is ~1M tokens, so we can pass very large
# documents verbatim; only beyond this do we fall back to the extraction notes.
COMPOSE_MAX_CHARS = 600_000

# Per-item compose/merge calls are independent network-bound LLM calls, so we run
# them concurrently (bounded) instead of sequentially. This keeps total ingest time
# well under the reverse-proxy timeout even for documents with many pages.
COMPOSE_CONCURRENCY = 8

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
    return _llm


def _windows(text: str) -> list[str]:
    """Split text into overlapping windows covering the whole document."""
    text = text.strip()
    if len(text) <= WINDOW_CHARS:
        return [text] if text else []
    out: list[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + WINDOW_CHARS])
        start += WINDOW_CHARS - WINDOW_OVERLAP
    return out


# ── Structured-output schemas ───────────────────────────────────────────────

class SourceSummary(BaseModel):
    title: str = Field(description="Concise human-readable title for this document.")
    summary: str = Field(description="A single sentence describing the document.")
    overview: str = Field(description=(
        "A THOROUGH markdown summary covering all major sections, points, and findings "
        "of the document — several paragraphs, with bullet lists where useful. Do not "
        "reduce it to a couple of sentences; capture the substance so nothing important "
        "is lost."
    ))


class ExtractedItem(BaseModel):
    name: str = Field(description="Name of the entity or concept.")
    kind: str = Field(description="Either 'entity' (a specific named thing) or 'concept' (a broad idea/topic/process).")
    summary: str = Field(description="One-line description used in the index.")
    coverage: str = Field(description=(
        "An EXHAUSTIVE checklist (markdown bullets) of every point the source makes about "
        "this item that the page must cover: definitions, key facts, figures, parameters, "
        "formulas, steps, examples, edge cases, and relationships to other items. This is a "
        "to-do list for the page writer — list everything so nothing is dropped. It does not "
        "need to be polished prose; completeness matters more than style."
    ))


class Extraction(BaseModel):
    items: list[ExtractedItem] = Field(description="Entities and a small number of broad concepts.")


# ── Pipeline steps ──────────────────────────────────────────────────────────

_SUMMARY_SYS = (
    "You are a meticulous knowledge-base editor. Summarize the given source text "
    "faithfully, completely, and concisely-but-thoroughly. Cover every major point; "
    "do not invent facts."
)

_EXTRACT_SYS = (
    "You IDENTIFY the pages a structured wiki should have for a document. A separate "
    "step will later write each page in full, so here you only name the items and list "
    "what each must cover — do NOT write polished prose.\n"
    "Extract two kinds of pages:\n"
    "- ENTITIES: specific named things (people, products, organizations, systems, "
    "places, datasets, tools). Include each distinct one that genuinely matters.\n"
    "- CONCEPTS: broad ideas, processes, or topics. Be SELECTIVE and CONSOLIDATING: "
    "prefer FEW, BROAD concept pages and fold related sub-topics into a single "
    "comprehensive page rather than creating many thin pages. Aim for roughly 2-6 "
    "concepts for a typical document.\n"
    "Crucially, the concepts and entities TOGETHER must cover the WHOLE document: every "
    "section, mechanism, and finding should fall under some page. For each item, fill "
    "'coverage' with an exhaustive checklist of everything the source says about it so "
    "the writer can be complete. Only include items supported by the source; never "
    "invent. Set 'kind' to exactly 'entity' or 'concept'."
)

_COMPOSE_SYS = (
    "You are writing ONE comprehensive wiki page about a single topic, using a source "
    "document. Your overriding goal is INFORMATION RETENTION: capture at least 90% of "
    "everything the source says that is relevant to this topic.\n"
    "- Include every definition, fact, figure, parameter, formula, step, assumption, "
    "example, caveat, and conclusion the source provides about the topic.\n"
    "- Preserve numbers, equations, and technical specifics EXACTLY; do not round or "
    "paraphrase them away.\n"
    "- Do NOT compress the material into a few sentences. Reproduce the substance in "
    "full, organized with markdown headings, paragraphs, and bullet/numbered lists.\n"
    "- Use ONLY information supported by the source; never invent. Where the source "
    "relates this topic to others, note the relationship.\n"
    "- Return ONLY the markdown body for the page (no YAML frontmatter, no code fences). "
    "Do not include a top-level '# Title' heading; start with the content."
)


def _summarize_window(text: str, filename: str) -> SourceSummary:
    llm = _get_llm().with_structured_output(SourceSummary)
    return llm.invoke([
        SystemMessage(content=_SUMMARY_SYS),
        HumanMessage(content=f"Source file: {filename}\n\n---\n{text}\n---"),
    ])


def _summarize_document(windows: list[str], filename: str) -> SourceSummary:
    if len(windows) == 1:
        return _summarize_window(windows[0], filename)
    # Map: summarize each window. Reduce: unify the partial overviews.
    partials = [_summarize_window(w, filename).overview for w in windows]
    combined = "\n\n---\n\n".join(f"[Part {i + 1}]\n{p}" for i, p in enumerate(partials))
    llm = _get_llm().with_structured_output(SourceSummary)
    return llm.invoke([
        SystemMessage(content=_SUMMARY_SYS),
        HumanMessage(content=(
            f"Source file: {filename}\nThe document was summarized in parts below. "
            f"Produce one unified, thorough summary covering ALL parts without dropping "
            f"information.\n\n{combined}"
        )),
    ])


def _extract_window(text: str, filename: str) -> list[ExtractedItem]:
    llm = _get_llm().with_structured_output(Extraction)
    result: Extraction = llm.invoke([
        SystemMessage(content=_EXTRACT_SYS),
        HumanMessage(content=f"Source file: {filename}\n\n---\n{text}\n---"),
    ])
    return result.items


def _extract_document(windows: list[str], filename: str) -> list[ExtractedItem]:
    if len(windows) == 1:
        return _extract_window(windows[0], filename)
    # Accumulate across windows, merging items that refer to the same thing so a
    # concept spanning multiple windows ends up on one rich page (no info dropped).
    acc: dict[tuple[str, str], ExtractedItem] = {}
    for w in windows:
        for item in _extract_window(w, filename):
            key = (_category_for(item.kind), wiki.slugify(item.name))
            if key in acc:
                acc[key].coverage = f"{acc[key].coverage}\n{item.coverage}"
            else:
                acc[key] = item
    return list(acc.values())


def _compose_messages(item: ExtractedItem, source_text: str, filename: str) -> list:
    return [
        SystemMessage(content=_COMPOSE_SYS),
        HumanMessage(content=(
            f"TOPIC: {item.name}\n"
            f"TOPIC TYPE: {item.kind}\n"
            f"ONE-LINE SUMMARY: {item.summary}\n\n"
            f"POINTS THIS PAGE MUST COVER (from a first pass — be exhaustive, and add "
            f"anything else relevant you find in the source):\n{item.coverage.strip()}\n\n"
            f"SOURCE DOCUMENT ({filename}):\n---\n{source_text}\n---\n\n"
            f"Now write the comprehensive page body for '{item.name}', extracting ALL "
            f"relevant information from the source above."
        )),
    ]


def _compose_bodies(items: list[ExtractedItem], source_text: str, filename: str) -> list[str]:
    """Compose a comprehensive page body for every item, concurrently.

    Each item gets its own generation call (with the whole source available) so the
    page can retain >=90% of the relevant material instead of a few sentences. The
    calls are independent, so we run them as a bounded-concurrency batch to keep
    wall-clock low. For documents too large to pass verbatim we fall back to the
    extraction checklist (no LLM call). Returns bodies aligned to `items`.
    """
    if not items:
        return []
    if not source_text or len(source_text) > COMPOSE_MAX_CHARS:
        return [it.coverage.strip() for it in items]
    llm = _get_llm()
    batch = [_compose_messages(it, source_text, filename) for it in items]
    results = llm.batch(batch, config={"max_concurrency": COMPOSE_CONCURRENCY})
    return [r.content.strip() for r in results]


_MERGE_SYS = (
    "You are updating an existing wiki page with a newly composed version built "
    "from a newly ingested source. CRITICAL RULES:\n"
    "- PRESERVE all existing information. Never delete or contradict prior content "
    "without explicit reason from the new source.\n"
    "- Integrate the new content thoroughly (merge, deduplicate, reconcile) so the "
    "page stays comprehensive and no detail from EITHER version is lost. Retain at "
    "least 90% of the substance of both; keep numbers, formulas, and specifics exact.\n"
    "- Return ONLY the updated markdown body (no YAML frontmatter, no code fences)."
)


def _merge_messages(existing_body: str, new_body: str, source_slug: str) -> list:
    return [
        SystemMessage(content=_MERGE_SYS),
        HumanMessage(content=(
            f"EXISTING PAGE BODY:\n{existing_body}\n\n"
            f"NEWLY COMPOSED CONTENT (from source '{source_slug}'):\n"
            f"{new_body}\n\n"
            "Return the full updated page body."
        )),
    ]


def _new_page_body(item: ExtractedItem, body: str, source_slug: str) -> str:
    return (
        f"# {item.name}\n\n"
        f"{body.strip()}\n\n"
        f"## Sources\n"
        f"- [[{wiki.SOURCES_DIR}/{source_slug}]]\n"
    )


def _category_for(kind: str) -> str:
    return "entities" if kind.strip().lower().startswith("entit") else "concepts"


def _write_new_page(item: ExtractedItem, category: str, slug: str, composed: str,
                    source_slug: str) -> None:
    wiki.write_page(category, slug, {
        "title": item.name,
        "type": category[:-1],  # 'entities' -> 'entity'
        "summary": item.summary,
        "sources": [source_slug],
        "updated": date.today().isoformat(),
    }, _new_page_body(item, composed, source_slug))


def _write_merged_page(item: ExtractedItem, category: str, slug: str, meta: dict,
                       merged_body: str, source_slug: str) -> None:
    # Ensure the source is linked even if the model omitted it.
    if f"{wiki.SOURCES_DIR}/{source_slug}" not in merged_body:
        merged_body += f"\n- [[{wiki.SOURCES_DIR}/{source_slug}]]\n"
    sources = meta.get("sources") or []
    if source_slug not in sources:
        sources.append(source_slug)
    meta.update({
        "title": meta.get("title", item.name),
        "type": category[:-1],
        "summary": meta.get("summary") or item.summary,
        "sources": sources,
        "updated": date.today().isoformat(),
    })
    wiki.write_page(category, slug, meta, merged_body)


def _upsert_items(items: list[ExtractedItem], composed_bodies: list[str],
                  source_slug: str) -> tuple[list[str], list[str]]:
    """Create new pages and merge into existing ones. Returns (created, updated).

    Merges into existing pages are independent LLM calls, so they are run as a
    bounded-concurrency batch rather than one at a time.
    """
    # Phase 1: classify each item as new vs existing (local filesystem reads only).
    plans = []  # (item, category, slug, composed, existing_or_None)
    for item, composed in zip(items, composed_bodies):
        category = _category_for(item.kind)
        slug = wiki.slugify(item.name)
        plans.append((item, category, slug, composed, wiki.read_page(category, slug)))

    # Phase 2: batch the merge LLM calls for the items whose page already exists.
    merge_targets = [p for p in plans if p[4] is not None]
    merged_bodies: dict[int, str] = {}
    if merge_targets:
        llm = _get_llm()
        batch = [_merge_messages(p[4][1], p[3], source_slug) for p in merge_targets]
        results = llm.batch(batch, config={"max_concurrency": COMPOSE_CONCURRENCY})
        for p, r in zip(merge_targets, results):
            merged_bodies[id(p)] = r.content.strip()

    # Phase 3: write everything (local filesystem writes only).
    created, updated = [], []
    for plan in plans:
        item, category, slug, composed, existing = plan
        try:
            if existing is None:
                _write_new_page(item, category, slug, composed, source_slug)
                created.append(f"{category}/{slug}")
            else:
                _write_merged_page(item, category, slug, existing[0],
                                   merged_bodies[id(plan)], source_slug)
                updated.append(f"{category}/{slug}")
        except Exception:  # one bad item shouldn't abort the whole ingest
            logger.exception("Failed to upsert wiki item %r", item.name)
    return created, updated


# ── Orchestrator ────────────────────────────────────────────────────────────

def ingest_document(original_filename: str, full_text: str) -> dict:
    """Run the full ingest pipeline for one document. Returns a result summary."""
    wiki.ensure_wiki()

    base_slug = wiki.slugify(original_filename.rsplit(".", 1)[0])
    source_slug = wiki.unique_source_slug(base_slug)
    wiki.write_source(source_slug, original_filename, full_text)

    # Process the ENTIRE document (no silent truncation) via overlapping windows.
    windows = _windows(full_text)
    if not windows:
        return {"source_slug": source_slug, "summary_title": original_filename,
                "pages_created": [], "pages_updated": [], "wiki_stats": wiki.stats()}

    summary = _summarize_document(windows, original_filename)
    wiki.write_page(
        "summaries",
        source_slug,
        {
            "title": summary.title,
            "type": "summary",
            "summary": summary.summary,
            "sources": [source_slug],
            "updated": date.today().isoformat(),
        },
        f"# {summary.title}\n\n{summary.overview.strip()}\n\n"
        f"## Source\n- [[{wiki.SOURCES_DIR}/{source_slug}]]\n",
    )

    items = _extract_document(windows, original_filename)
    composed_bodies = _compose_bodies(items, full_text, original_filename)
    created, updated = _upsert_items(items, composed_bodies, source_slug)

    wiki.regenerate_index()
    wiki.append_log(
        "ingest",
        summary.title,
        f"source: {source_slug} · {len(created)} new page(s), {len(updated)} updated.",
    )

    return {
        "source_slug": source_slug,
        "summary_title": summary.title,
        "pages_created": created,
        "pages_updated": updated,
        "wiki_stats": wiki.stats(),
    }
