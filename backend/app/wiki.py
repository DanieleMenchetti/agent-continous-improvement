"""Filesystem layer for the LLM-maintained markdown wiki.

Implements the three-layer architecture from karpathy's "LLM wiki" guidelines
(https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):

    1. Raw sources  -> sources/<slug>.md   (immutable; never rewritten)
    2. The wiki     -> summaries/, entities/, concepts/ + index.md
    3. The schema   -> CLAUDE.md            (conventions, written once)

Plus an append-only log.md. Ingestion is *incremental and non-destructive*:
existing pages are merged/extended, never deleted or replaced wholesale, and
log.md / sources/ are append-only.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

from .config import settings

# Wiki page categories (subdirectories holding generated markdown pages).
CATEGORIES = ("summaries", "entities", "concepts")
SOURCES_DIR = "sources"

CLAUDE_MD = """# Wiki Schema & Conventions

This directory is an **LLM-maintained wiki**, built incrementally from ingested
documents. It follows a three-layer architecture:

1. **`sources/`** — raw, immutable extracted text of each ingested document. Never edited.
2. **Wiki pages** — generated markdown the agent owns and keeps up to date:
   - `summaries/` — one page per source document.
   - `entities/`  — pages about specific named things (people, products, orgs, systems).
   - `concepts/`  — pages about ideas, processes, and topics.
3. **`CLAUDE.md`** (this file) — conventions for how the wiki is structured.

## Special files
- **`index.md`** — auto-generated catalog of every page, grouped by category, with
  one-line summaries. Regenerated on every ingest.
- **`log.md`** — append-only chronological record. Each ingest adds a line like
  `## [YYYY-MM-DD] ingest | <title>`.

## Page format
Every generated page begins with YAML frontmatter:

```yaml
---
title: Human Readable Title
type: entity | concept | summary
summary: One-line description used in index.md.
sources: [source-slug-a, source-slug-b]
updated: YYYY-MM-DD
---
```

followed by markdown. Cross-references use `[[category/slug]]` wiki-links.

## Ingestion rule
Ingestion is **additive**. Existing pages are merged with new information; prior
content is preserved. Sources and the log are append-only.
"""


def _root() -> Path:
    return Path(settings.wiki_dir)


def ensure_wiki() -> None:
    """Create the wiki skeleton (dirs + CLAUDE.md/index.md/log.md) if missing."""
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    (root / SOURCES_DIR).mkdir(exist_ok=True)
    for cat in CATEGORIES:
        (root / cat).mkdir(exist_ok=True)

    claude = root / "CLAUDE.md"
    if not claude.exists():
        claude.write_text(CLAUDE_MD, encoding="utf-8")

    log = root / "log.md"
    if not log.exists():
        log.write_text("# Ingestion Log\n\nAppend-only record of wiki ingests.\n", encoding="utf-8")

    index = root / "index.md"
    if not index.exists():
        index.write_text("# Wiki Index\n\n_(empty — ingest a document to populate)_\n", encoding="utf-8")


def slugify(name: str) -> str:
    """Filesystem-safe slug: lowercase, hyphenated, alnum only."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "untitled"


# ── Pages (frontmatter + body) ──────────────────────────────────────────────

def _page_file(category: str, slug: str) -> Path:
    return _root() / category / f"{slug}.md"


def read_page(category: str, slug: str) -> tuple[dict, str] | None:
    """Return (frontmatter, body) for a page, or None if it doesn't exist."""
    path = _page_file(category, slug)
    if not path.exists():
        return None
    return _parse_frontmatter(path.read_text(encoding="utf-8"))


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            return (meta if isinstance(meta, dict) else {}), parts[2].lstrip("\n")
    return {}, text


def write_page(category: str, slug: str, frontmatter: dict, body: str) -> None:
    """Write a page with YAML frontmatter. Overwrites the named page only."""
    front = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    content = f"---\n{front}\n---\n\n{body.strip()}\n"
    _page_file(category, slug).write_text(content, encoding="utf-8")


def list_pages(category: str) -> list[str]:
    d = _root() / category
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.md"))


# ── Sources (immutable, append-only) ────────────────────────────────────────

def source_exists(slug: str) -> bool:
    return (_root() / SOURCES_DIR / f"{slug}.md").exists()


def unique_source_slug(base_slug: str) -> str:
    """Return a free source slug, suffixing -2, -3, ... so we never overwrite."""
    if not source_exists(base_slug):
        return base_slug
    n = 2
    while source_exists(f"{base_slug}-{n}"):
        n += 1
    return f"{base_slug}-{n}"


def write_source(slug: str, original_filename: str, text: str) -> None:
    body = (
        f"---\n"
        f"title: {original_filename}\n"
        f"type: source\n"
        f"slug: {slug}\n"
        f"ingested: {date.today().isoformat()}\n"
        f"---\n\n"
        f"# Raw source: {original_filename}\n\n"
        f"> Immutable extracted text. Do not edit.\n\n"
        f"{text.strip()}\n"
    )
    (_root() / SOURCES_DIR / f"{slug}.md").write_text(body, encoding="utf-8")


# ── Log (append-only) ───────────────────────────────────────────────────────

def append_log(kind: str, title: str, details: str = "") -> None:
    line = f"\n## [{date.today().isoformat()}] {kind} | {title}\n"
    if details:
        line += f"{details}\n"
    with (_root() / "log.md").open("a", encoding="utf-8") as f:
        f.write(line)


# ── Index (regenerated programmatically from page frontmatter) ──────────────

_CATEGORY_TITLES = {
    "summaries": "Source Summaries",
    "entities": "Entities",
    "concepts": "Concepts",
}


def regenerate_index() -> None:
    """Rebuild index.md as a catalog of all pages with their one-line summaries."""
    lines = ["# Wiki Index", "", f"_Last updated: {date.today().isoformat()}_", ""]
    total = 0
    for cat in CATEGORIES:
        slugs = list_pages(cat)
        if not slugs:
            continue
        lines.append(f"## {_CATEGORY_TITLES.get(cat, cat.title())}")
        lines.append("")
        for slug in slugs:
            page = read_page(cat, slug)
            meta = page[0] if page else {}
            title = meta.get("title", slug)
            summary = meta.get("summary", "")
            entry = f"- [[{cat}/{slug}]] **{title}**"
            if summary:
                entry += f" — {summary}"
            lines.append(entry)
            total += 1
        lines.append("")
    if total == 0:
        lines.append("_(empty — ingest a document to populate)_")
    (_root() / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def stats() -> dict:
    return {
        "sources": len(list(( _root() / SOURCES_DIR).glob("*.md"))) if (_root() / SOURCES_DIR).exists() else 0,
        **{cat: len(list_pages(cat)) for cat in CATEGORIES},
    }
