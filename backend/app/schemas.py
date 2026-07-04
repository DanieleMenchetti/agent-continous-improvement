"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---- Chat ----
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    qa_id: int
    answer: str
    used_feedback: list[str] = []  # snippets of retrieved expert feedback


# ---- Feedback ----
class FeedbackCreate(BaseModel):
    content: str = Field(..., min_length=1)
    rating: int | None = Field(default=None, ge=1, le=5)
    is_correction: bool = True


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    qa_pair_id: int
    content: str
    rating: int | None
    is_correction: bool
    created_at: datetime
    updated_at: datetime


# ---- Config ----
class SoulPrompt(BaseModel):
    prompt: str = ""


class WikiPageEdit(BaseModel):
    content: str = Field(..., min_length=1)


# ---- Wiki health check (lint) ----
class LintFinding(BaseModel):
    check: str            # e.g. "orphan", "contradiction", "coverage_gap"
    severity: str         # "error" | "warning" | "info"
    title: str
    detail: str
    pages: list[str] = []  # affected "category/slug" refs
    suggestion: str = ""


class LintReport(BaseModel):
    generated: str                    # ISO date the check was run
    healthy: bool                     # True when no errors/warnings were found
    counts: dict[str, int]            # {"error": n, "warning": n, "info": n}
    findings: list[LintFinding] = []
    stats: dict = {}


# ---- Q&A pairs (expert review view) ----
class QAPairOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    session_id: str | None
    created_at: datetime
    feedback: FeedbackOut | None = None
