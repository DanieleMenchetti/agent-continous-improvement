"""Config endpoints: Agent Soul prompt and document ingestion."""
import io
import re

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from .. import vectorstore
from ..database import get_db
from ..models import KeyValueSetting
from ..schemas import SoulPrompt

router = APIRouter(prefix="/api/config", tags=["config"])

SOUL_KEY = "agent_soul"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ── Agent Soul ────────────────────────────────────────────────────────────────

@router.get("/soul", response_model=SoulPrompt)
def get_soul(db: Session = Depends(get_db)) -> SoulPrompt:
    row = db.get(KeyValueSetting, SOUL_KEY)
    return SoulPrompt(prompt=row.value if row else "")


@router.put("/soul", response_model=SoulPrompt)
def put_soul(body: SoulPrompt, db: Session = Depends(get_db)) -> SoulPrompt:
    row = db.get(KeyValueSetting, SOUL_KEY)
    if row is None:
        row = KeyValueSetting(key=SOUL_KEY, value=body.prompt)
        db.add(row)
    else:
        row.value = body.prompt
    db.commit()
    return SoulPrompt(prompt=row.value)


# ── Document ingestion ────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks of ~CHUNK_SIZE chars."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


@router.post("/documents")
async def upload_document(file: UploadFile, db: Session = Depends(get_db)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    try:
        from pypdf import PdfReader
    except ImportError:
        raise HTTPException(status_code=500, detail="pypdf not installed.")

    raw = await file.read()
    reader = PdfReader(io.BytesIO(raw))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    chunks = _chunk_text(full_text)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename)
    count = vectorstore.upsert_document_chunks(safe_name, chunks)

    return {"filename": file.filename, "pages": len(reader.pages), "chunks_indexed": count}
