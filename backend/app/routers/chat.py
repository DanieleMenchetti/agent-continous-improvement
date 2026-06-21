"""Customer-facing chat endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import agent
from ..database import get_db
from ..models import QAPair
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # 1. Agent answers, grounded in expert feedback retrieved via RAG.
    answer, retrieved = agent.answer_question(req.question)

    # 2. Persist the Q&A pair so the expert can later review it.
    pair = QAPair(question=req.question, answer=answer, session_id=req.session_id)
    db.add(pair)
    db.commit()
    db.refresh(pair)

    return ChatResponse(
        qa_id=pair.id,
        answer=answer,
        used_feedback=[h["metadata"].get("feedback", "") for h in retrieved],
    )
