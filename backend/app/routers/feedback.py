"""Expert feedback endpoints. Every create/update is mirrored to the vector DB."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import vectorstore
from ..database import get_db
from ..models import Feedback, QAPair
from ..schemas import FeedbackCreate, FeedbackOut

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


def _sync_to_vectorstore(pair: QAPair, fb: Feedback) -> None:
    vectorstore.upsert_feedback(
        feedback_id=fb.id,
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

    # Mirror into the vector store so future answers benefit from it.
    _sync_to_vectorstore(pair, fb)
    return fb


@router.delete("/{qa_id}", status_code=204)
def delete_feedback(qa_id: int, db: Session = Depends(get_db)):
    pair = db.get(QAPair, qa_id)
    if pair is None or pair.feedback is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    fb_id = pair.feedback.id
    db.delete(pair.feedback)
    db.commit()
    vectorstore.delete_feedback(fb_id)
