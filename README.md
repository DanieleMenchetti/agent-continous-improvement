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
   - API docs: http://localhost:8000/docs

## API summary

| Method | Path                  | Purpose                                   |
|--------|-----------------------|-------------------------------------------|
| POST   | `/api/chat`           | Ask the agent; stores the Q&A pair        |
| GET    | `/api/qa`             | List Q&A pairs (`?only_unreviewed=true`)  |
| GET    | `/api/qa/{id}`        | Get one Q&A pair                          |
| PUT    | `/api/feedback/{qa_id}` | Create/update feedback (syncs to vector DB) |
| DELETE | `/api/feedback/{qa_id}` | Remove feedback (and its vector entry)  |

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

## Notes

- Tables are auto-created on startup. For production migrations, add Alembic.
- One feedback entry per Q&A pair (updating overwrites it, in both DBs).
- CORS is open (`*`) for convenience; restrict it before exposing publicly.
