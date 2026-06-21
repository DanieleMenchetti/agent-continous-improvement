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


# ---- Q&A pairs (expert review view) ----
class QAPairOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    session_id: str | None
    created_at: datetime
    feedback: FeedbackOut | None = None
