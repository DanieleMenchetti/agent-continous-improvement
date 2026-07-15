"""Customer-facing chat endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import agent, history
from ..database import get_db
from ..models import KeyValueSetting, QAPair
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    # Prepend the Agent Soul prompt if one has been configured.
    soul_row = db.get(KeyValueSetting, "agent_soul")
    soul_prompt = soul_row.value if soul_row else ""

    # Build the conversation-history block for this session: recent turns verbatim
    # plus a rolling LLM summary of older ones (empty for a brand-new session).
    convo_history = history.build_history(db, req.session_id)

    # 1. Agent answers, grounded in wiki knowledge retrieved via RAG, guided by the
    #    configured Agent Soul, and aware of the conversation so far.
    answer, retrieved = agent.answer_question(
        req.question, soul_prompt=soul_prompt, history=convo_history
    )

    # 2. Persist the Q&A pair so the expert can later review it.
    pair = QAPair(question=req.question, answer=answer, session_id=req.session_id)
    db.add(pair)
    db.commit()
    db.refresh(pair)

    # Distinct page titles that grounded the answer (chunks may repeat a page).
    used_context: list[str] = []
    for h in retrieved:
        title = h["metadata"].get("title", "")
        if title and title not in used_context:
            used_context.append(title)

    return ChatResponse(qa_id=pair.id, answer=answer, used_context=used_context)
