# Agent Continous Improvement

An agent that continuously improves from a human domain expert's feedback.

- **Customers** chat with the agent. Every question/answer pair is stored.
- **A domain expert** reviews those pairs and writes feedback on the ones worth correcting.
- Each piece of feedback is embedded into a **vector database**.
- Before answering, the agent **retrieves relevant past feedback (RAG)** and grounds its reply in it.

## Architecture

```
                ┌──────────────┐         ┌──────────────┐
  Customer ───▶ │  chat.html   │         │ feedback.html│ ◀── Domain expert
                └──────┬───────┘         └──────┬───────┘
                       │  /api (nginx proxy)    │
                       ▼                        ▼
                ┌───────────────────────────────────────┐
                │  FastAPI backend                       │
                │   • LangGraph agent (retrieve→generate)│
                │   • Gemini 2.5 Flash (chat + embeds)   │
                └───────┬───────────────────┬────────────┘
                        ▼                   ▼
                ┌───────────────┐   ┌────────────────┐
                │ PostgreSQL    │   │ Chroma (vector)│
                │ Q&A + feedback│   │ expert feedback│
                └───────────────┘   └────────────────┘
```

Services (`docker-compose.yml`): `postgres`, `chroma`, `backend`, `frontend`.

## How the feedback loop works

1. `POST /api/chat` → LangGraph runs `retrieve` (vector search over feedback) then
   `generate` (Gemini answers, grounded in retrieved guidance). The Q&A pair is
   saved to Postgres.
2. The expert opens **Expert review**, sees the Q&A pairs, and submits feedback via
   `PUT /api/feedback/{qa_id}`.
3. On every create/update, the feedback is **upserted into Chroma** — keyed by the
   original question's embedding, with the expert's guidance as the retrievable text.
4. Future similar questions retrieve that guidance, so the agent improves over time.

## Run it

1. Get a Gemini API key: https://aistudio.google.com/apikey
2. Configure the key:
   ```bash
   cp .env.example .env
   # edit .env and set GOOGLE_API_KEY
   ```
3. Start everything:
   ```bash
   docker compose up --build
   ```
4. Open:
   - Customer chat: http://localhost:8080/chat.html
   - Expert review: http://localhost:8080/feedback.html
   - Wiki reader: http://localhost:8080/wiki.html
   - API docs: http://localhost:8000/docs

## API summary

| Method | Path                  | Purpose                                   |
|--------|-----------------------|-------------------------------------------|
| POST   | `/api/chat`           | Ask the agent; stores the Q&A pair        |
| GET    | `/api/qa`             | List Q&A pairs (`?only_unreviewed=true`)  |
| GET    | `/api/qa/{id}`        | Get one Q&A pair                          |
| PUT    | `/api/feedback/{qa_id}` | Create/update feedback (syncs to vector DB) |
| DELETE | `/api/feedback/{qa_id}` | Remove feedback (and its vector entry)  |
| POST   | `/api/config/documents` | Upload a PDF; ingests it into the wiki    |
| GET    | `/api/config/wiki`      | Catalog of wiki pages (slug/title/summary) + stats |
| GET    | `/api/config/wiki/{category}/{slug}` | Raw markdown of one wiki page |

## Document ingestion → markdown wiki

Uploading a PDF (Config page or `POST /api/config/documents`) runs a fixed-pipeline
**ingestion agent** that builds a persistent markdown wiki on the server, following
[karpathy's LLM-wiki guidelines](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
The wiki lives in a Docker volume (`wikidata` → `/data/wiki`, configurable via `WIKI_DIR`)
with three layers:

```
/data/wiki/
  CLAUDE.md          # schema & conventions (written once)
  index.md           # auto-regenerated catalog of all pages
  log.md             # append-only ingest log
  sources/           # immutable raw extracted text, one per document
  summaries/         # one summary page per source
  entities/          # pages about named things (merged across sources)
  concepts/          # pages about ideas/topics (merged across sources)
```

The pipeline: **summarize** the source → **extract** entities/concepts → **write or
non-destructively merge** each page → **regenerate** `index.md` → **append** to `log.md`.
Ingestion is **incremental**: re-ingesting or adding new documents extends existing
pages and never deletes prior wiki data (sources and the log are append-only; existing
pages are merged, not replaced). The wiki is not yet wired into chat retrieval — the
uploaded text is still chunk-indexed into Chroma for chat as before.

The wiki can be read as a book at **`/wiki.html`**: a contents sidebar (chapters grouped
by Concepts / Entities / Summaries / Sources), a paper-style reading surface, page-flip
Previous/Next (also ←/→ keys), and clickable `[[category/slug]]` cross-references. It
renders the markdown client-side and deep-links each page via the URL hash.

## Configuration

Backend settings (env vars, see `backend/app/config.py`):

| Var               | Default                      |
|-------------------|------------------------------|
| `GOOGLE_API_KEY`  | (required)                   |
| `CHAT_MODEL`      | `gemini-2.5-flash`           |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001`  |
| `RAG_TOP_K`       | `4`                          |
| `DATABASE_URL`    | Postgres DSN                 |
| `CHROMA_HOST/PORT`| `chroma` / `8000`            |
| `WIKI_DIR`        | `/data/wiki`                 |

## Notes

- Tables are auto-created on startup. For production migrations, add Alembic.
- One feedback entry per Q&A pair (updating overwrites it, in both DBs).
- CORS is open (`*`) for convenience; restrict it before exposing publicly.
