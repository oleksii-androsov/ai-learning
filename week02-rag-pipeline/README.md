# Week 2 — RAG Pipeline from Scratch

## What we're building

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a document. Instead of sending the whole document to the model every time, it indexes the document upfront and only retrieves the relevant sections at query time.

## Architecture

```
Indexing (once):
Document → paragraphs → chunks → Bedrock Titan embeddings → FAISS index (saved to disk)

Retrieval (per question):
Question → Bedrock Titan embedding → FAISS search → relevant chunks
Relevant chunks + conversation history → Claude → answer
```

## Getting started

1. Clone the repo and follow the setup in the root README
2. Add AWS credentials to `.env` (see `.env.example`)
3. Place your document at `week02-rag-pipeline/document.txt`
4. Run: `python3 week02-rag-pipeline/rag.py`

The index is built automatically on first run and reused on subsequent runs. If the document changes, the index rebuilds automatically (detected via MD5 hash).

## Day-by-day progress

### Day 5 — Indexer and retriever as separate scripts
- Connected to AWS Bedrock, generated first embeddings with Amazon Titan Text Embeddings V2
- Split document into word-count-based chunks with overlap
- Built FAISS index and saved to disk (`index.faiss` + `chunks.json`)
- Built separate retriever script: question → embedding → FAISS search → Claude answer

### Day 6 — Single pipeline script with conversation history
- Wired indexer and retriever into a single `rag.py`
- Added index freshness check: loads from disk if index exists, builds from scratch if not
- Added MD5 file hash check: detects if document has changed since index was built and rebuilds automatically
- Added conversation history: follow-up questions have context from previous answers

### Day 7 — Smarter chunking and relevance filtering
- Replaced word-count chunking with paragraph-aware chunking: chunks respect document structure, paragraphs are never split mid-sentence
- Tuned `CHUNK_SIZE` to 150 words based on actual document paragraph lengths (producing 9 focused chunks vs 3 large blobs)
- Added relevance threshold (`RELEVANCE_THRESHOLD = 0.5`): FAISS results below the threshold are discarded rather than passed to Claude
- Added general knowledge fallback: if no relevant chunks are found, Claude answers from general knowledge with a clear note to the user

## Key architectural decisions

- **AWS Bedrock + Titan Embeddings V2** — enterprise-grade embeddings, consistent with AWS-first stack. 512 dimensions with normalization.
- **Paragraph-aware chunking** — respects document structure over arbitrary word boundaries. Produces more semantically coherent chunks.
- **MD5 hash for cache invalidation** — prevents stale index serving when document is updated under the same filename. Production pattern.
- **Relevance threshold** — avoids passing irrelevant context to Claude, which can confuse responses. Real production concern.
- **General knowledge fallback** — makes the assistant useful beyond the document scope while being transparent about the source of the answer.
- **FAISS IndexFlatIP** — Inner Product similarity, equivalent to cosine similarity for normalized vectors. Exact search, appropriate for small-to-medium corpora.

## What I'd do differently

- **File hash per chunk** — currently the whole index rebuilds if anything in the document changes. A smarter system would only re-embed changed sections.
- **Sentence-level chunking** — using an NLP library (spaCy, NLTK) to split on sentence boundaries rather than paragraph boundaries for even more precise retrieval.
- **Configurable document path** — accept the document path as a command-line argument rather than hardcoding it.
- **Persistent conversation history** — save and reload conversation history across sessions.
- **Week 3** — replace FAISS with Amazon OpenSearch Serverless for a production-grade vector store.
