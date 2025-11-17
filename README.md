# Enterprise RAG Confluence Webhooks

FastAPI service that ingests Confluence page creation + update webhooks, fetches the latest page body, generates embeddings with LlamaIndex + the `bge-m3` model running on Ollama, and stores vectors inside Postgres with the pgvector extension.

## Features
- `/webhook/confluence` endpoint accepts Confluence page events and schedules background ingestion.
- Confluence REST fetch + HTML to text conversion keeps metadata (title, space, version, author).
- Custom LlamaIndex embedding class that calls the local Ollama endpoint for `bge-m3` embeddings.
- Vectors persisted to Postgres via `PGVectorStore` with deterministic node IDs for idempotent upserts.
- Docker Compose recipe for `pgvector/pgvector:pg16` plus `.env.example` documenting all configuration keys.

## Prerequisites
- Python 3.11+
- Running Ollama instance with the `bge-m3` model pulled (`ollama pull bge-m3`).
- Confluence Cloud API token with read access to spaces you plan to ingest.

## Setup
1. Copy the sample env file and adjust credentials:
   ```bash
   cp .env.example .env
   ```
2. Install dependencies into your virtualenv:
   ```bash
   pip install -r requirements.txt
   ```
3. Start pgvector locally (optional if you already have Postgres):
   ```bash
   docker compose up -d
   ```

## Running the service
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
The server exposes `/webhook/confluence` and `/health`. Point the Confluence webhook subscription to `https://<host>/webhook/confluence` and allow Atlassian's IPs.

## Config highlights
| Variable | Description |
| --- | --- |
| `CONFLUENCE_BASE_URL` | Base Atlassian domain, e.g., `https://example.atlassian.net`. |
| `CONFLUENCE_USERNAME` / `CONFLUENCE_API_TOKEN` | Credentials used for REST fetches. |
| `CONFLUENCE_SPACE_WHITELIST` | Optional comma list of space keys that should be indexed. |
| `OLLAMA_BASE_URL` | Ollama instance root (defaults to `http://localhost:11434`). |
| `EMBEDDING_MODEL_NAME` | Embedding model passed to Ollama (defaults to `bge-m3`). |
| `DATABASE_URL_ASYNC` | Async connection string (e.g., `postgresql+asyncpg://…`). Provide this **or** `DATABASE_URL`. |
| `DATABASE_URL` | Sync psycopg connection string. Provide this **or** `DATABASE_URL_ASYNC`; the missing one is auto-derived. |
| `DATABASE_SCHEMA` | Postgres schema the vector table lives in (defaults to `public`). |
| `VECTOR_COLLECTION` | Table name used by `PGVectorStore`. |

## Triggering ingestion
Confluence will send payloads containing `eventType` (e.g., `page_created`, `page_updated`). The webhook handler acknowledges immediately (HTTP 202) and performs ingestion asynchronously. Logs describe each page's ingestion lifecycle.

## Next steps
- Add webhook secret validation (X-Atlassian-Webhook-Identifier) for higher security.
- Extend the service with a retrieval API or scheduled re-sync jobs.


##clear cache using
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete