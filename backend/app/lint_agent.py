"""Wiki health-check ("lint") agent.

Implements the periodic health check from karpathy's "LLM wiki" guidelines
(https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):

    > Periodically, ask the LLM to health-check the wiki. Look for: contradictions
    > between pages, stale claims that newer sources have superseded, orphan pages
    > with no inbound links, important concepts mentioned but lacking their own
    > page, missing cross-references, data gaps that could be filled with a web
    > search.

The lint is a hybrid pass:

  * **Structural checks** (deterministic, no LLM) run first and are cheap/exact:
    frontmatter schema integrity, broken `[[wiki-links]]`, and orphan pages with
    no inbound links.
  * **Semantic checks** (one LLM call over the corpus) cover the judgement calls:
    contradictions, stale claims, coverage gaps, missing cross-references, and
    data gaps worth investigating.

The agent never edits the wiki — it only *reports*. Each finding carries a
severity so the UI can notify the user when something actually needs attention.
"""
from __future__ import annotations

import logging
import re
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from . import wiki
from .config import settings

logger = logging.getLogger(__name__)

# Frontmatter every generated page is expected to carry (see wiki.CLAUDE_MD).
REQUIRED_FRONTMATTER = ("title", "type", "summary", "sources", "updated")

# Only concepts/entities form the cross-referenced knowledge graph, so those are
# the categories where an "orphan" (no inbound links) is meaningful. Summaries are
# per-document leaves and sources are raw, so they are excluded from that check.
GRAPH_CATEGORIES = ("concepts", "entities")

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Corpus budget for the semantic pass. Gemini has a very large context window, but
# we still cap per-page and total size so a huge wiki can't blow the request up.
PAGE_BODY_CAP = 8_000
CORPUS_CHAR_CAP = 400_000

# Semantic-check identifiers the LLM is allowed to emit.
SEMANTIC_CHECKS = ("contradiction", "stale", "coverage_gap", "missing_xref", "data_gap")

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
    return _llm


# ── Loading the wiki into memory ─────────────────────────────────────────────

class _Page:
    __slots__ = ("category", "slug", "meta", "body")

    def __init__(self, category: str, slug: str, meta: dict, body: str):
        self.category = category
        self.slug = slug
        self.meta = meta
        self.body = body

    @property
    def ref(self) -> str:
        return f"{self.category}/{self.slug}"

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.slug)


def _load_pages() -> list[_Page]:
    """Read every generated page (concepts/entities/summaries) with body + meta."""
    pages: list[_Page] = []
    for cat in wiki.CATEGORIES:
        for slug in wiki.list_pages(cat):
            page = wiki.read_page(cat, slug)
            if page is None:
                continue
            meta, body = page
            pages.append(_Page(cat, slug, meta, body))
    return pages


def _all_refs() -> set[str]:
    """Every addressable page ref, including immutable sources (link targets)."""
    refs = {f"{cat}/{slug}" for cat in wiki.CATEGORIES for slug in wiki.list_pages(cat)}
    refs |= {f"{wiki.SOURCES_DIR}/{slug}" for slug in wiki.list_pages(wiki.SOURCES_DIR)}
    return refs


def _resolve_link(ref: str, valid_refs: set[str], slug_index: dict[str, str]) -> str | None:
    """Resolve a raw `[[...]]` target to a canonical `category/slug`, or None.

    Accepts both `category/slug` and bare `slug` links (the latter resolved by
    looking the slug up across categories).
    """
    ref = ref.strip()
    if "/" in ref:
        cat, slug = ref.split("/", 1)
        canonical = f"{cat.strip()}/{slug.strip()}"
        return canonical if canonical in valid_refs else None
    return slug_index.get(ref)


# ── Structural checks (deterministic) ────────────────────────────────────────

def _structural_findings(pages: list[_Page]) -> list[dict]:
    findings: list[dict] = []
    valid_refs = _all_refs()
    # Map bare slug -> canonical ref (skip ambiguous slugs shared across categories).
    slug_counts: dict[str, list[str]] = {}
    for ref in valid_refs:
        slug = ref.split("/", 1)[1]
        slug_counts.setdefault(slug, []).append(ref)
    slug_index = {s: refs[0] for s, refs in slug_counts.items() if len(refs) == 1}

    inbound: dict[str, int] = {p.ref: 0 for p in pages}

    for p in pages:
        # Schema integrity — flag missing required frontmatter.
        missing = [f for f in REQUIRED_FRONTMATTER if not p.meta.get(f)]
        if missing:
            findings.append(_finding(
                "schema", "warning",
                f"Incomplete frontmatter on “{p.title}”",
                f"Missing field(s): {', '.join(missing)}.",
                [p.ref],
                "Edit the page and add the missing frontmatter so it stays indexable.",
            ))

        # Broken links + inbound-link tally (for orphan detection).
        for raw in _WIKILINK_RE.findall(p.body):
            target = _resolve_link(raw, valid_refs, slug_index)
            if target is None:
                findings.append(_finding(
                    "broken_link", "error",
                    f"Broken link on “{p.title}”",
                    f"Links to `[[{raw.strip()}]]`, which does not resolve to any page.",
                    [p.ref],
                    "Fix the target slug, or create the page it should point to.",
                ))
            elif target != p.ref and target in inbound:
                inbound[target] += 1

    # Orphan pages — graph pages nothing else links to.
    for p in pages:
        if p.category in GRAPH_CATEGORIES and inbound.get(p.ref, 0) == 0:
            findings.append(_finding(
                "orphan", "warning",
                f"Orphan page: “{p.title}”",
                "No other page links to this one, so readers can't reach it by navigating.",
                [p.ref],
                "Add a `[[" + p.ref + "]]` cross-reference from a related page.",
            ))

    return findings


# ── Semantic checks (LLM) ────────────────────────────────────────────────────

class SemanticFinding(BaseModel):
    check: str = Field(description=(
        "One of: 'contradiction' (two pages disagree), 'stale' (a claim a newer "
        "source has superseded), 'coverage_gap' (an important concept mentioned but "
        "with no dedicated page), 'missing_xref' (a page discusses another existing "
        "page without linking it), 'data_gap' (an open question or missing fact worth "
        "a web search)."
    ))
    severity: str = Field(description="'error', 'warning', or 'info'.")
    title: str = Field(description="A short one-line headline for the finding.")
    detail: str = Field(description="A concise explanation of the issue and why it matters.")
    pages: list[str] = Field(default_factory=list, description=(
        "The 'category/slug' refs of the pages this finding concerns (may be empty "
        "for a coverage_gap or data_gap about content that does not yet exist)."
    ))
    suggestion: str = Field(default="", description="A concrete next step to resolve it.")


class SemanticReport(BaseModel):
    findings: list[SemanticFinding] = Field(default_factory=list)


_SEMANTIC_SYS = (
    "You are a meticulous knowledge-base editor performing a health check ('lint') "
    "on an LLM-maintained wiki. You are given every page's title, category/slug, "
    "last-updated date, and body. Inspect the corpus as a whole and report problems.\n\n"
    "Look ONLY for these issue types and set 'check' to the exact identifier:\n"
    "- contradiction: two pages state facts that conflict. Cite BOTH pages.\n"
    "- stale: a claim that a more recently-updated page or source has superseded or "
    "made outdated. Cite the stale page (and the newer one if relevant).\n"
    "- coverage_gap: an important concept, entity, or term that pages mention "
    "repeatedly but which has no dedicated page of its own. Cite the page(s) that "
    "mention it; name the missing page in the title.\n"
    "- missing_xref: a page clearly discusses another topic that DOES have its own "
    "page, but does not link to it with a [[category/slug]] wiki-link. Cite both.\n"
    "- data_gap: an open question, unknown, or obviously missing fact that a web "
    "search could fill, keeping the wiki healthy as it grows.\n\n"
    "Rules: Only report issues genuinely supported by the provided content — never "
    "invent. Prefer a few high-value findings over many trivial ones. Use severity "
    "'error' for contradictions and hard breakages, 'warning' for staleness, and "
    "'info' for coverage gaps, missing cross-references, and data gaps. Reference "
    "pages ONLY by their exact 'category/slug' as given. If the wiki looks healthy, "
    "return an empty list."
)


def _build_corpus(pages: list[_Page]) -> str:
    manifest = "\n".join(f"- {p.ref} — {p.title}" for p in pages)
    blocks: list[str] = [f"PAGE MANIFEST ({len(pages)} pages):\n{manifest}\n"]
    total = len(blocks[0])
    for p in pages:
        body = p.body.strip()
        if len(body) > PAGE_BODY_CAP:
            body = body[:PAGE_BODY_CAP] + "\n…[truncated]"
        block = (
            f"\n=== {p.ref} === (updated: {p.meta.get('updated', 'unknown')})\n"
            f"Title: {p.title}\n"
            f"Summary: {p.meta.get('summary', '')}\n\n"
            f"{body}\n"
        )
        if total + len(block) > CORPUS_CHAR_CAP:
            blocks.append("\n…[remaining pages omitted from this lint pass]")
            break
        blocks.append(block)
        total += len(block)
    return "".join(blocks)


def _semantic_findings(pages: list[_Page]) -> list[dict]:
    if not settings.google_api_key:
        return [_finding(
            "config", "info", "Semantic checks skipped",
            "No GOOGLE_API_KEY is configured, so contradiction/staleness/coverage "
            "checks were not run. Only structural checks were performed.",
            [], "Set GOOGLE_API_KEY to enable the full health check.",
        )]

    valid_refs = _all_refs()
    corpus = _build_corpus(pages)
    llm = _get_llm().with_structured_output(SemanticReport)
    result: SemanticReport = llm.invoke([
        SystemMessage(content=_SEMANTIC_SYS),
        HumanMessage(content=f"Here is the full wiki corpus to health-check:\n\n{corpus}"),
    ])

    findings: list[dict] = []
    for f in result.findings:
        check = f.check.strip().lower()
        if check not in SEMANTIC_CHECKS:
            continue
        # Keep only page refs that actually exist (drops hallucinated targets).
        refs = [r.strip() for r in f.pages if r.strip() in valid_refs]
        # For checks that assert something about specific pages, a finding whose
        # every referenced page is invalid is almost certainly a hallucination.
        if check in ("contradiction", "stale", "missing_xref") and not refs:
            continue
        severity = f.severity.strip().lower()
        if severity not in ("error", "warning", "info"):
            severity = "info"
        findings.append(_finding(check, severity, f.title.strip(),
                                 f.detail.strip(), refs, f.suggestion.strip()))
    return findings


# ── Report assembly ──────────────────────────────────────────────────────────

_SEVERITY_RANK = {"error": 0, "warning": 1, "info": 2}


def _finding(check: str, severity: str, title: str, detail: str,
             pages: list[str], suggestion: str = "") -> dict:
    return {
        "check": check,
        "severity": severity,
        "title": title,
        "detail": detail,
        "pages": pages,
        "suggestion": suggestion,
    }


def lint_wiki() -> dict:
    """Run the full health check and return a structured report.

    The return value is JSON-serializable and matches the ``LintReport`` schema.
    Structural checks always run; semantic checks run when an API key is present
    and degrade gracefully (a single 'error' finding) if the LLM call fails.
    """
    wiki.ensure_wiki()
    pages = _load_pages()

    findings = _structural_findings(pages)
    if pages:
        try:
            findings += _semantic_findings(pages)
        except Exception:
            logger.exception("Semantic lint pass failed")
            findings.append(_finding(
                "semantic_error", "warning", "Semantic checks could not complete",
                "The LLM health-check pass failed to run. Structural checks are still "
                "shown below. Check the server logs and try again.",
                [], "Retry the health check; if it persists, verify the model config.",
            ))

    findings.sort(key=lambda f: (_SEVERITY_RANK.get(f["severity"], 3), f["check"]))
    counts = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    return {
        "generated": date.today().isoformat(),
        "healthy": counts["error"] == 0 and counts["warning"] == 0,
        "counts": counts,
        "findings": findings,
        "stats": wiki.stats(),
    }
