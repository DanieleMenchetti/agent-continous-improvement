"""Lightweight ingestion of domain-expert feedback into the wiki.

Where the PDF pipeline (``ingest_agent.py``) is deliberately heavy — windowing the
whole document, then summarize + extract + a dedicated compose pass per item +
per-item merges (many LLM calls) — expert feedback is short and already curated.
So ingesting it is intentionally light: a SINGLE LLM call turns one corrected Q&A
into a concise, self-contained knowledge note, which is written as one page in the
dedicated ``feedback`` wiki category.

The page slug is derived from the Q&A pair id, so re-submitting feedback for the
same pair updates its page in place (matching the one-feedback-per-pair model),
and deleting the feedback removes the page.
"""
from __future__ import annotations

import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from . import wiki, wiki_sync
from .config import settings

logger = logging.getLogger(__name__)

CATEGORY = "feedback"

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


# ── Structured output for the single polish call ─────────────────────────────

class FeedbackNote(BaseModel):
    title: str = Field(description=(
        "A short, human-readable title naming the topic this guidance is about "
        "(roughly 4-8 words). Not a full sentence."
    ))
    summary: str = Field(description="A single-line description used in the wiki index.")
    note: str = Field(description=(
        "A concise, self-contained knowledge note in markdown that captures the "
        "expert's corrected guidance as durable, reusable knowledge — written so a "
        "future agent can apply it, NOT as a reply to one customer. Do not include a "
        "top-level '# Title' heading or YAML frontmatter; start with the content."
    ))


_SYS = (
    "You maintain a knowledge base built from a domain expert's feedback on an AI "
    "agent's answers. Turn ONE piece of feedback into a concise, durable knowledge "
    "note.\n"
    "- Capture the corrected guidance as a general rule or fact, not a one-off reply "
    "to this exact customer.\n"
    "- Be faithful and brief: do NOT invent details beyond the expert's guidance.\n"
    "- Where the expert's guidance and the original answer conflict, the expert's "
    "guidance is authoritative.\n"
    "- Return only the note body as markdown (no frontmatter, no top-level heading)."
)


def _compose(question: str, answer: str, feedback_content: str, is_correction: bool) -> FeedbackNote:
    """The single LLM call: polish the corrected Q&A into a knowledge note."""
    stance = (
        "The expert is CORRECTING the agent's answer."
        if is_correction
        else "The expert is refining or confirming the agent's answer."
    )
    llm = _get_llm().with_structured_output(FeedbackNote)
    return llm.invoke([
        SystemMessage(content=_SYS),
        HumanMessage(content=(
            f"{stance}\n\n"
            f"CUSTOMER QUESTION:\n{question}\n\n"
            f"AGENT'S ORIGINAL ANSWER:\n{answer}\n\n"
            f"EXPERT GUIDANCE:\n{feedback_content}\n\n"
            "Write the knowledge note."
        )),
    ])


def _fallback_note(question: str, feedback_content: str) -> FeedbackNote:
    """Used only if the LLM call fails, so feedback is never lost — verbatim, no LLM."""
    q = question.strip()
    title = (q[:57] + "…") if len(q) > 58 else (q or "Expert feedback")
    return FeedbackNote(
        title=title,
        summary="Expert guidance (stored verbatim).",
        note=(
            f"**When a customer asks something like:** {q}\n\n"
            f"**Expert guidance:** {feedback_content.strip()}"
        ),
    )


def _slug(qa_id: int) -> str:
    return f"qa-{qa_id}"


# ── Public API ───────────────────────────────────────────────────────────────

def ingest_feedback(
    *,
    qa_id: int,
    question: str,
    answer: str,
    feedback_content: str,
    is_correction: bool,
    rating: int | None,
) -> dict:
    """Write (or update) the wiki page for one piece of expert feedback."""
    wiki.ensure_wiki()
    slug = _slug(qa_id)

    try:
        note = _compose(question, answer, feedback_content, is_correction)
    except Exception:  # never drop feedback because a network/LLM call failed
        logger.exception("Feedback polish failed for qa_id=%s; storing verbatim", qa_id)
        note = _fallback_note(question, feedback_content)

    footer_bits = ["correction" if is_correction else "refinement"]
    if rating is not None:
        footer_bits.append(f"expert rated original answer {rating}/5")
    body = (
        f"# {note.title}\n\n"
        f"{note.note.strip()}\n\n"
        f"---\n\n"
        f"_Source: expert feedback on Q&A #{qa_id} · {' · '.join(footer_bits)}._\n"
    )

    frontmatter: dict = {
        "title": note.title,
        "type": "feedback",
        "summary": note.summary,
        "sources": [slug],
        "is_correction": is_correction,
        "updated": date.today().isoformat(),
    }
    if rating is not None:
        frontmatter["rating"] = rating

    existed = wiki.page_exists(CATEGORY, slug)
    wiki.write_page(CATEGORY, slug, frontmatter, body)
    wiki.regenerate_index()
    wiki_sync.index_page(CATEGORY, slug)  # keep RAG in sync with the new/updated page
    wiki.append_log(
        "feedback",
        note.title,
        f"page: {CATEGORY}/{slug} ({'updated' if existed else 'new'}).",
    )
    logger.info("Ingested expert feedback for qa_id=%s into wiki (%s)", qa_id, slug)
    return {"category": CATEGORY, "slug": slug, "title": note.title, "created": not existed}


def delete_feedback_page(qa_id: int) -> bool:
    """Remove the wiki page for a Q&A pair's feedback. Returns False if absent."""
    slug = _slug(qa_id)
    removed = wiki.delete_page(CATEGORY, slug)
    if removed:
        wiki.regenerate_index()
        wiki_sync.remove_page(CATEGORY, slug)  # drop it from RAG too
        logger.info("Removed feedback wiki page for qa_id=%s", qa_id)
    return removed
