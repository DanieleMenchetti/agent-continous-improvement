"""Expert feedback endpoints. Every create/update is ingested into the wiki."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import feedback_ingest
from ..database import get_db
from ..models import Feedback, QAPair
from ..schemas import FeedbackCreate, FeedbackOut

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def _sync_to_wiki(pair: QAPair, fb: Feedback) -> None:
    feedback_ingest.ingest_feedback(
        qa_id=pair.id,
        question=pair.question,
        answer=pair.answer,
        feedback_content=fb.content,
        is_correction=fb.is_correction,
        rating=fb.rating,
    )


@router.put("/{qa_id}", response_model=FeedbackOut)
def upsert_feedback(qa_id: int, payload: FeedbackCreate, db: Session = Depends(get_db)):
    """Create or update the feedback for a Q&A pair (one feedback per pair)."""
    pair = db.get(QAPair, qa_id)
    if pair is None:
        raise HTTPException(status_code=404, detail="Q&A pair not found")

    fb = pair.feedback
    if fb is None:
        fb = Feedback(qa_pair_id=qa_id)
        db.add(fb)

    fb.content = payload.content
    fb.rating = payload.rating
    fb.is_correction = payload.is_correction
    db.commit()
    db.refresh(fb)
    db.refresh(pair)

    # Capture the guidance as a durable note in the wiki knowledge base.
    _sync_to_wiki(pair, fb)
    return fb


@router.delete("/{qa_id}", status_code=204)
def delete_feedback(qa_id: int, db: Session = Depends(get_db)):
    pair = db.get(QAPair, qa_id)
    if pair is None or pair.feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    db.delete(pair.feedback)
    db.commit()
    feedback_ingest.delete_feedback_page(qa_id)
