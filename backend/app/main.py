"""FastAPI application entrypoint."""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import wiki, wiki_sync
from .database import Base, engine
from .routers import chat, config, feedback, qa

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Self-Improving Domain Agent")

# Allow the static frontend (served separately by nginx) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    # Create tables if they don't exist (simple bootstrap; use Alembic in prod).
    Base.metadata.create_all(bind=engine)
    # Ensure the wiki skeleton exists and the vector store reflects it (best-effort:
    # if the store is empty but the wiki has pages, e.g. a fresh Chroma volume, build
    # it so RAG works from the first request).
    wiki.ensure_wiki()
    wiki_sync.ensure_indexed()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(chat.router)
app.include_router(qa.router)
app.include_router(feedback.router)
app.include_router(config.router)
