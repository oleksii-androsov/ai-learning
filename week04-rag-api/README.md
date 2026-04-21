# Week 4 — RAG API: Deploy with FastAPI

## What we're building

Wrapping the Week 3 RAG pipeline in a FastAPI web service so it can be called over HTTP — from a browser, a frontend, or any other client. The RAG logic is unchanged; the interface shifts from a terminal loop to an API endpoint.

## Architecture

```
Client (curl / Streamlit)
    ↓ HTTP POST /ask {"question": "..."}
FastAPI (api.py)
    ↓ expand query → embed → search → generate
Pinecone + AWS Bedrock + Claude
    ↓
{"answer": "...", "sources_found": true}
```

## Getting started

1. Clone the repo and follow the setup in the root README
2. Add AWS and Pinecone credentials to `.env` (see `.env.example`)
3. Place your document at `week03-rag-v2/document.txt`
4. Run: `uvicorn week04-rag-api.api:app --reload`
5. Test: `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -H "X-API-Key: your-key" -d '{"question": "What is this document about?"}'`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` — confirms server is running |
| POST | `/ask` | Accepts `{"question": "..."}`, returns `{"answer": "...", "sources_found": true/false}`. Requires `X-API-Key` header. |

## Day-by-day progress

### Day 17 — Streamlit frontend
- Built a chat UI with `streamlit_app.py` — text input, message history, spinner while waiting for response
- Deployed on the same EC2 instance as the API, accessible at port 8501
- UI maintains conversation display via `st.session_state` — previous messages stay visible across interactions
- Each question sent to the API independently (stateless) — conversation history in UI only, not in AI context
- Graceful error handling for timeouts and unreachable API
- Port 8501 opened in EC2 security group restricted to own IP

### Day 16 — Datadog observability (unplanned)
- Installed Datadog Agent on EC2 — infrastructure metrics (CPU, memory, disk, network) flowing automatically
- Replaced `print()` with Python `logging` module and custom `JsonFormatter` — every log line is now a JSON object
- Configured Agent to tail `server.log`, added Grok parser to extract `method`, `path`, `status_code`, `duration_ms`
- Created log-based metric `rag.api.request.count` grouped by `status_code`
- Instrumented with `ddtrace-run` — zero code changes, automatic spans for FastAPI, Anthropic, Bedrock, and Pinecone
- Built dashboard combining infrastructure metrics, log-based metrics, and APM latency
- Created 3 monitors: CPU threshold, p95 latency, error rate

### Day 15 — EC2 deployment + API key authentication
- Added API key authentication to `/ask` — requests without a valid `X-API-Key` header get 401
- Deployed to AWS EC2 (t3.micro, Ubuntu 24.04, eu-central-1) — API now accessible at a real public IP
- Server runs via `nohup` so it survives SSH disconnection, logs to `server.log`
- Security group restricted: SSH and port 8000 open to own IP only

### Day 14 — FastAPI wrapper
- Wrapped the Week 3 RAG pipeline in a FastAPI web server
- `/ask` endpoint: accepts a question as JSON, runs the full RAG pipeline (query expansion → Pinecone retrieval → Claude), returns answer as JSON
- Clients (Bedrock, Anthropic, Pinecone) loaded once at startup via FastAPI `lifespan` — not recreated per request
- Added `if __name__ == "__main__":` guard to `week03-rag-v2/rag.py` so it can be imported without running its interactive loop
- Stateless for now: each request is answered independently, no conversation history between requests

## Key architectural decisions

- **Stateless requests** — no server-side conversation history in this iteration. Each `/ask` call is independent. Three options for adding history: client-side (Streamlit sends full history with each request), server-side in-memory (dict keyed by session ID, lost on restart), or persistent store (DynamoDB/Redis). Will iterate toward Option 3.
- **Clients created at startup, not per request** — Pinecone, Bedrock, and Anthropic clients involve network connections and credential resolution. Creating them once and reusing them avoids that overhead on every request.
- **RAG logic imported from week03** — no code duplication. `sys.path` used to import across folders with hyphenated names. In a production codebase this logic would live in a shared package.
- **API key auth via `X-API-Key` header** — static secret stored in `.env`, checked on every `/ask` request via a FastAPI dependency. Simple and effective for a private API. `/health` is intentionally left public for monitoring.

## What I'd do differently

- **Add conversation history** — iterate through client-side, then server-side session, then DynamoDB persistence
- **Add a shared package** — extract RAG functions into a proper Python package rather than using `sys.path` manipulation
- **Add request validation** — minimum question length, rate limiting, auth token
