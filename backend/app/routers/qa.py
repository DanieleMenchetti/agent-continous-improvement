"""Q&A listing for the domain expert review page."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import QAPair
from ..schemas import QAPairOut

router = APIRouter(prefix="/api/qa", tags=["qa"])


@router.get("", response_model=list[QAPairOut])
def list_qa_pairs(
    db: Session = Depends(get_db),
    only_unreviewed: bool = Query(False, description="Only pairs without feedback"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    stmt = select(QAPair).order_by(QAPair.created_at.desc())
    pairs = db.scalars(stmt.limit(limit).offset(offset)).all()
    if only_unreviewed:
        pairs = [p for p in pairs if p.feedback is None]
    return pairs


@router.get("/{qa_id}", response_model=QAPairOut)
def get_qa_pair(qa_id: int, db: Session = Depends(get_db)):
    pair = db.get(QAPair, qa_id)
    if pair is None:
        raise HTTPException(status_code=404, detail="Q&A pair not found")
    return pair
