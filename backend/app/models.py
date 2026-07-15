"""ORM models: customer Q&A pairs, expert feedback, and app settings."""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class KeyValueSetting(Base):
    """Generic key/value store for runtime configuration (e.g. agent_soul)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")


class QAPair(Base):
    """A single customer question and the agent's answer."""

    __tablename__ = "qa_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    feedback: Mapped["Feedback | None"] = relationship(
        back_populates="qa_pair",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ConversationSummary(Base):
    """Rolling LLM summary of the older turns of a chat session.

    Recent turns are sent to the agent verbatim; once a turn scrolls out of that
    window it is folded into ``summary`` exactly once. ``summarized_up_to`` records
    the highest :class:`QAPair` id already absorbed, so each old turn is summarized a
    single time instead of re-summarizing the whole history on every request.
    """

    __tablename__ = "conversation_summaries"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Highest QAPair.id already folded into ``summary`` (0 = nothing folded yet).
    summarized_up_to: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Feedback(Base):
    """Expert guidance on a Q&A pair. Ingested into the wiki (and thus into RAG)."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    qa_pair_id: Mapped[int] = mapped_column(
        ForeignKey("qa_pairs.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    # The expert's correction / improved answer / guidance.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional 1..5 quality rating of the original answer.
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Whether this feedback marks the original answer as good or needing change.
    is_correction: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    qa_pair: Mapped["QAPair"] = relationship(back_populates="feedback")
