"""Conversation memory for the chat agent.

A chat session is a sequence of :class:`QAPair` rows (question + answer) keyed by
``session_id``. Feeding every past turn to the LLM on each request would grow the
prompt without bound, so this module produces a compact history block:

* the last ``settings.history_recent_turns`` turns, verbatim, so recent detail is
  never lost;
* a single rolling summary of everything older than that window.

The summary is built **incrementally**. Each session keeps a
:class:`ConversationSummary` row recording the summary text and the highest QAPair
id already absorbed (``summarized_up_to``). When turns scroll out of the recent
window we fold only those *new* older turns into the existing summary with one LLM
call — we never re-summarize the whole history. A session's summary is therefore
touched at most once per request, and only when a turn has actually aged out.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from . import agent
from .config import settings
from .models import ConversationSummary, QAPair

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = """You maintain a running summary of a customer-support \
conversation. You are given the summary so far (possibly empty) and the next batch \
of older turns that have scrolled out of the live window. Produce an updated summary \
that folds the new turns into the existing one.

Keep it concise but preserve anything future replies may depend on: the customer's \
goal, decisions and commitments made, unresolved questions, and concrete facts \
(names, numbers, identifiers, preferences). Drop pleasantries and redundancy. Write \
plain prose in the third person; output only the updated summary."""


def _format_turns(pairs: list[QAPair]) -> str:
    blocks = [f"Customer: {p.question}\nAgent: {p.answer}" for p in pairs]
    return "\n\n".join(blocks)


def _fold_into_summary(prior_summary: str, new_older: list[QAPair]) -> str:
    """Ask the LLM to merge ``new_older`` turns into ``prior_summary``."""
    prior = prior_summary.strip() or "(no summary yet)"
    user_content = (
        f"SUMMARY SO FAR:\n{prior}\n\n"
        f"NEW OLDER TURNS TO FOLD IN:\n{_format_turns(new_older)}"
    )
    messages = [
        SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = agent._get_llm().invoke(messages)
    return (response.content or "").strip()


def _update_summary(db: Session, session_id: str, older: list[QAPair]) -> str:
    """Return the summary covering all of ``older``, folding in any new turns.

    Only the turns with an id beyond ``summarized_up_to`` are sent to the LLM, so the
    cost of maintaining the summary is proportional to newly-aged-out turns, not to
    the length of the whole conversation.
    """
    row = db.get(ConversationSummary, session_id)
    covered = row.summarized_up_to if row else 0

    new_older = [p for p in older if p.id > covered]
    if not new_older:
        return row.summary if row else ""

    updated = _fold_into_summary(row.summary if row else "", new_older)
    high_id = older[-1].id
    if row is None:
        row = ConversationSummary(
            session_id=session_id, summary=updated, summarized_up_to=high_id
        )
        db.add(row)
    else:
        row.summary = updated
        row.summarized_up_to = high_id
    db.commit()
    logger.info(
        "Folded %d older turn(s) into summary for session '%s'",
        len(new_older),
        session_id,
    )
    return updated


def build_history(db: Session, session_id: str | None) -> str:
    """Build the conversation-history block for the current request.

    Returns an empty string when there is no prior context (no session, or a brand
    new session). Otherwise returns the rolling summary (if any) followed by the most
    recent turns verbatim.
    """
    if not session_id:
        return ""

    pairs = (
        db.query(QAPair)
        .filter(QAPair.session_id == session_id)
        .order_by(QAPair.created_at, QAPair.id)
        .all()
    )
    if not pairs:
        return ""

    recent_n = max(settings.history_recent_turns, 0)
    older = pairs[:-recent_n] if recent_n else pairs
    recent = pairs[-recent_n:] if recent_n else []

    summary = _update_summary(db, session_id, older) if older else ""

    parts: list[str] = []
    if summary:
        parts.append(f"SUMMARY OF EARLIER CONVERSATION:\n{summary}")
    if recent:
        parts.append(f"RECENT MESSAGES:\n{_format_turns(recent)}")
    return "\n\n".join(parts)
