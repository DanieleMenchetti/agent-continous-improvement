# Agent Continous Improvement

An agent that continuously improves from a human domain expert's feedback.

- **Customers** chat with the agent. Every question/answer pair is stored.
- **A domain expert** reviews those pairs and writes feedback on the ones worth correcting.
- Each piece of feedback is ingested into a persistent **markdown knowledge base (wiki)**
  as a concise, durable guidance note.
- Uploaded PDFs are ingested into the same wiki.

## Architecture

```
                ┌──────────────┐         ┌──────────────┐
  Customer ───▶ │  chat.html   │         │ feedback.html│ ◀── Domain expert
                └──────┬───────┘         └──────┬───────┘
                       │  /api (nginx proxy)    │
                       ▼                        ▼
                ┌───────────────────────────────────────┐
                │  FastAPI backend                       │
                │   • LangGraph agent (generate)         │
                │   • Gemini 2.5 Flash (chat + ingest)   │
                └───────┬───────────────────┬────────────┘
                        ▼                   ▼
                ┌───────────────┐   ┌─────────────────────┐
                │ PostgreSQL    │   │ Wiki (markdown files)│
                │ Q&A + feedback│   │ /data/wiki volume    │
                └───────────────┘   └─────────────────────┘
```

Services (`docker-compose.yml`): `postgres`, `backend`, `frontend`.

## How the feedback loop works

1. `POST /api/chat` → the LangGraph agent runs `generate` (Gemini answers, guided by
   the configured Agent Soul). The Q&A pair is saved to Postgres.
2. The expert opens **Expert review**, sees the Q&A pairs, and submits feedback via
   `PUT /api/feedback/{qa_id}`.
3. On every create/update, a single LLM call polishes the correction into a durable
   guidance note, written to the wiki under the `feedback/` category (one page per
   Q&A pair; updating overwrites it, deleting removes it).

> The wiki is a human-readable knowledge base built from expert feedback and ingested
> documents. It is **not yet wired into chat answering** — the agent currently answers
> from the Agent Soul and its own knowledge (there is no RAG/retrieval step).

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
| PUT    | `/api/feedback/{qa_id}` | Create/update feedback (ingested into the wiki) |
| DELETE | `/api/feedback/{qa_id}` | Remove feedback (and its wiki page)     |
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
  feedback/          # expert-feedback guidance notes (one per corrected Q&A pair)
```

The PDF pipeline: **summarize** the source → **extract** entities/concepts → **write or
non-destructively merge** each page → **regenerate** `index.md` → **append** to `log.md`.
Ingestion is **incremental**: re-ingesting or adding new documents extends existing
pages and never deletes prior wiki data (sources and the log are append-only; existing
pages are merged, not replaced). Expert feedback takes a much lighter path — a single
LLM call that writes one `feedback/` page per Q&A pair. The wiki is a knowledge base
for people to read; it is not consumed by the chat agent.

The wiki can be read as a book at **`/wiki.html`**: a contents sidebar (chapters grouped
by Concepts / Entities / Expert Feedback / Summaries / Sources), a paper-style reading surface, page-flip
Previous/Next (also ←/→ keys), and clickable `[[category/slug]]` cross-references. It
renders the markdown client-side and deep-links each page via the URL hash.

## Configuration

Backend settings (env vars, see `backend/app/config.py`):

| Var               | Default                      |
|-------------------|------------------------------|
| `GOOGLE_API_KEY`  | (required)                   |
| `CHAT_MODEL`      | `gemini-2.5-flash`           |
| `DATABASE_URL`    | Postgres DSN                 |
| `WIKI_DIR`        | `/data/wiki`                 |

## Notes

- Tables are auto-created on startup. For production migrations, add Alembic.
- One feedback entry per Q&A pair (updating overwrites it, along with its wiki page).
- CORS is open (`*`) for convenience; restrict it before exposing publicly.
