# Webhook + Embeddings Architecture

## Flow overview
1. Confluence emits a webhook whenever a page is created or updated. Atlassian delivers a POST request to `/webhook/confluence` with the page ID and event type.
2. Our FastAPI application validates the event, acknowledges immediately, and hands processing to a background task queue (FastAPI `BackgroundTasks`) so webhook calls return quickly.
3. The background task fetches the page body from the Confluence REST API using the provided page ID.
4. The page body is converted to plain text and wrapped in a LlamaIndex `Document` with metadata that captures page URL, space key, author, and version.
5. LlamaIndex routes the cleaned text through a custom `OllamaBgeM3Embedding` implementation that calls the local Ollama server's `/api/embeddings` endpoint (model `bge-m3`).
6. The resulting vectors are persisted via `PGVectorStore` into Postgres (running the `pgvector/pgvector:16` image). Each chunk is upserted via deterministic node IDs so re-indexing the same page overwrites previous vectors.
7. Downstream retrieval pipelines can now query the pgvector table using LlamaIndex or plain SQL + `ivfflat`/`hnsw` indexes.

## Key components
- `app/config.py`: centralizes environment variables (Confluence credentials, Ollama endpoint, embedding model, Postgres URL, chunk sizes).
- `app/confluence.py`: helper to fetch page contents + metadata via the Confluence Cloud REST API.
- `app/embeddings.py`: lightweight LlamaIndex embedding class that calls Ollama and exposes both sync/async embedding hooks.
- `app/vector_store.py`: configures the `PGVectorStore` instance + helper to upsert documents.
- `app/webhook.py`: FastAPI router exposing `/webhook/confluence`, validates payloads, and schedules page processing.
- `docker-compose.yaml`: spins up `pgvector/pgvector:16` with pgvector extension enabled, persistent storage, and a healthcheck.
- `.env.example`: documents configuration keys for Confluence, embeddings, Ollama, and Postgres.

## Processing logic
```
POST /webhook/confluence
  -> ensure event is page_created|page_updated
  -> respond 202 immediately
  -> background task: process_page(page_id)
        fetch Confluence page JSON
        flatten storage.body to plain text
        chunk via LlamaIndex text splitter
        embed with OllamaBgeM3Embedding
        upsert into PGVectorStore (namespace = confluence)
```

## Future extensions
- Add signature validation for Confluence webhooks (HMAC secret) once configured.
- Add retry/backoff queue (Redis/ Celery) for resilient ingestion.
- Mirror page deletions by removing corresponding vectors.
- Layer a retrieval API (FastAPI route) for downstream question answering.
