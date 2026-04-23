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

## Deploying with Terraform

Prerequisites: AWS account, Terraform installed, AWS credentials configured.

**1. Create your variables file**
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```
Edit `terraform.tfvars` and paste your SSH public key:
```bash
ssh-keygen -y -f your-key.pem
```

**2. Provision the infrastructure**
```bash
terraform init
terraform apply
```
Takes about 15 seconds. Note the `public_ip` output.

**3. Wait for bootstrap to complete (~4 minutes)**

The instance runs `user_data.sh` on first boot — installs Python, clones the repo, sets up the venv, configures systemd. Monitor progress:
```bash
ssh -i your-key.pem ubuntu@<public_ip>
cat /var/log/user_data.log | tail -20
```
Bootstrap is done when you see the systemd symlinks created at the end of the log.

**4. Copy secrets onto the instance**
```bash
scp -i your-key.pem .env ubuntu@<public_ip>:~/ai-learning/.env
scp -i your-key.pem week03-rag-v2/document.txt ubuntu@<public_ip>:~/ai-learning/week03-rag-v2/document.txt
```

**5. Start the services**
```bash
ssh -i your-key.pem ubuntu@<public_ip>
sudo systemctl start rag-api rag-streamlit
```

**6. Verify**
```bash
curl -s http://localhost:8000/health
# → {"status":"ok"}
```

Streamlit UI is available at `http://<public_ip>:8501`

**7. Update CI/CD** (if using GitHub Actions)

Copy the `update_github_secret` output from `terraform apply` and run it:
```bash
gh secret set EC2_HOST --body <public_ip>
```

**Tear down when done:**
```bash
terraform destroy
```

> **Note:** Secrets are not automated. `.env` and `document.txt` must be copied manually after each fresh `terraform apply`. Production deployments would use AWS Secrets Manager for this.

## Running locally

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your credentials
3. Place your document at `week03-rag-v2/document.txt`
4. `pip install -r week04-rag-api/requirements.txt`
5. `uvicorn week04-rag-api.api:app --reload`
6. Test: `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -H "X-API-Key: your-key" -d '{"question": "What is this document about?"}'`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` — confirms server is running |
| POST | `/ask` | Accepts `{"question": "..."}`, returns `{"answer": "...", "sources_found": true/false}`. Requires `X-API-Key` header. |

## Day-by-day progress

### Day 18 — File upload, conversation history, CI/CD, Terraform
- `/upload` endpoint: accepts PDF or TXT via multipart form, re-indexes Pinecone without restarting the server
- Conversation history: Streamlit sends full message history with each `/ask` request; Claude sees prior turns in context
- Switched from `nohup` to systemd for service management — more reliable, survives reboots, restarts on failure
- CI/CD via GitHub Actions: `validate` job checks syntax on push/PR; `deploy` job SSH-es into EC2, pulls latest code, does `systemctl restart`, and verifies with a health check — all triggered on push to main
- First Terraform template (`terraform/`) — defines EC2 instance, security group, and key pair as code; `user_data.sh` bootstraps a fresh instance end-to-end; `terraform destroy` tears everything down cleanly

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

- **Client-side conversation history** — Streamlit sends the full message history with each `/ask` request. Claude sees prior turns without the server storing any state. Simple and stateless; the tradeoff is history is lost if the browser tab closes. Next iteration: DynamoDB for persistent sessions.
- **Stateless requests** — each `/ask` call is self-contained. Three options for server-side history: in-memory dict keyed by session ID (lost on restart), or DynamoDB/Redis for persistence. Currently using client-side (Option 1).
- **Clients created at startup, not per request** — Pinecone, Bedrock, and Anthropic clients involve network connections and credential resolution. Creating them once and reusing them avoids that overhead on every request.
- **RAG logic imported from week03** — no code duplication. `sys.path` used to import across folders with hyphenated names. In a production codebase this logic would live in a shared package.
- **API key auth via `X-API-Key` header** — static secret stored in `.env`, checked on every `/ask` request via a FastAPI dependency. Simple and effective for a private API. `/health` is intentionally left public for monitoring.

## What I'd do differently

- **DynamoDB for conversation history** — client-side history works but is lost on tab close; next step is persistent sessions keyed by session ID
- **Add a shared package** — extract RAG functions into a proper Python package rather than using `sys.path` manipulation
- **Add request validation** — minimum question length, rate limiting, input sanitisation
- **AWS Secrets Manager for secrets** — `.env` and `document.txt` require manual copying after each `terraform apply`; production deployments would pull secrets from Secrets Manager in `user_data.sh`
