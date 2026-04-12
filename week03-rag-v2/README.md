# Week 3 — RAG v2: Production Vector Store

## What we're building

An upgraded RAG pipeline that replaces the local FAISS index with Pinecone — a managed, cloud-hosted vector database. The core pipeline logic is identical to Week 2, but the vector store is now persistent, scalable, and accessible without rebuilding on every run.

## Architecture

```
Indexing (once, or when document changes):
Document → chunks → Bedrock Titan embeddings → Pinecone index (cloud-hosted)

Retrieval (per question):
Question → Bedrock Titan embedding → Pinecone similarity search → relevant chunks
Relevant chunks + conversation history → Claude → answer
```

## Getting started

1. Clone the repo and follow the setup in the root README
2. Add AWS and Pinecone credentials to `.env` (see `.env.example`)
3. Place your document at `week03-rag-v2/document.txt`
4. Run: `python3 week03-rag-v2/rag.py`

The Pinecone index is created automatically on first run. Document changes are detected via MD5 hash — the index rebuilds only when needed.

## Day-by-day progress

### Day 9 — Replace FAISS with Pinecone
- Swapped local FAISS index for Pinecone serverless vector database
- Index and chunk text stored together in Pinecone as vector + metadata — no separate `chunks.json` needed
- Index creation handled in code via `get_or_create_index()` — reproducible setup, no manual console steps
- Tuned `RELEVANCE_THRESHOLD` to 0.3 and `TOP_K` to 3 based on observed Pinecone similarity scores for this document
- All Week 2 features retained: paragraph chunking, conversation history, general knowledge fallback, hash-based cache invalidation

## Key architectural decisions

- **Pinecone over FAISS** — persistent, managed, no in-memory rebuilds. Vectors survive restarts. Scales to millions of vectors without code changes.
- **Pinecone over OpenSearch Serverless** — identical concepts, but Pinecone has a free tier suitable for learning and prototyping. OpenSearch Serverless is the recommended choice for enterprise deployments in regulated industries (FSI, healthcare) where data must stay within your AWS account and VPC.
- **Metadata for chunk text** — Pinecone stores original text alongside vectors as metadata. Eliminates the need for a separate chunk store. Returned directly with query results.
- **`upsert` for indexing** — Pinecone's upsert replaces existing vectors by ID on rebuild. Clean, idempotent — running twice produces the same result.
- **Threshold and TOP_K tuning** — Titan embeddings score conversational queries lower than keyword-matched ones. Threshold of 0.3 and TOP_K of 3 found empirically to balance precision and recall for this document type.

## What I'd do differently

- **Query expansion** — rewrite vague user questions into multiple specific ones before retrieval. Significantly improves recall for high-level questions.
- **Automatic eval on new documents** — generate test questions from a new document using Claude, then immediately validate retrieval quality.
- **Dynamic threshold** — calibrate per document rather than hardcoding. A relative threshold (top result must be X% better than average) would generalise better.
- **OpenSearch Serverless** — for a real enterprise deployment, replace Pinecone with OpenSearch Serverless in eu-central-1 for data sovereignty and hybrid search capability.
