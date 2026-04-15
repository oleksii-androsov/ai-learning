# Architecture Decision Log — Week 3 RAG v2

Each entry follows the ADR format: **Context → Decision → Alternatives → Tradeoffs**

---

## ADR-001: Pinecone over FAISS for vector storage

**Context**
The Week 2 pipeline used FAISS — a local, in-memory vector index. It works, but the index is lost every time the script stops and has to be rebuilt by re-embedding all chunks. This means API calls and wait time on every run, even when the document hasn't changed.

**Decision**
Replace FAISS with Pinecone, a managed cloud-hosted vector database. Vectors persist between runs. The index survives restarts.

**Alternatives considered**
- Keep FAISS, save index to disk with `faiss.write_index()` — avoids cloud dependency but still runs locally, doesn't scale
- Amazon OpenSearch Serverless — production-grade, stays within AWS, supports hybrid search (vector + keyword). Recommended for enterprise deployments, but has no free tier suitable for learning
- Weaviate, Qdrant — capable alternatives, but Pinecone has the clearest free tier and the most straightforward Python SDK

**Tradeoffs**
- Pinecone free tier is limited to one index and low query volume — fine for learning, not for production
- Adds an external cloud dependency and another API key to manage
- For a real FSI or healthcare deployment: OpenSearch Serverless in eu-central-1 is the right answer (data sovereignty, VPC isolation, no third-party dependency)

---

## ADR-002: Chunk text stored as Pinecone metadata

**Context**
In the FAISS pipeline, chunk text was stored separately in `chunks.json`. After retrieval, the code looked up chunks by index position. Two files to keep in sync — fragile if one gets out of date.

**Decision**
Store chunk text directly in Pinecone as metadata alongside each vector. Retrieval returns the text immediately with the search results — no separate lookup needed.

**Alternatives considered**
- Keep a separate `chunks.json` — familiar from Week 2, but adds a second file that can go stale
- Store chunk IDs and look up text from the original document — adds complexity and requires re-chunking at query time

**Tradeoffs**
- Pinecone metadata has size limits (40KB per record). Long chunks could hit this ceiling in production
- For very large documents, storing text externally (S3 + chunk ID) and doing a secondary lookup is safer
- For this use case (short document, short chunks), metadata is simpler and cleaner

---

## ADR-003: MD5 hash for cache invalidation

**Context**
Building the index is expensive — it requires embedding every chunk, which means one Bedrock API call per chunk. Re-running this on every script start would be slow and wasteful.

**Decision**
Hash the document file with MD5 on every run. Save the hash to disk after indexing. On the next run, compare the current hash to the saved one — rebuild only if they differ.

**Alternatives considered**
- File modification timestamp — simpler, but timestamps can be wrong (copied files, timezone issues, `touch`)
- Always rebuild — simplest code, but impractical for any document longer than a few pages
- Semantic change detection (diff the chunks) — overkill; any file change should trigger a rebuild anyway

**Tradeoffs**
- MD5 has known collision vulnerabilities, but for cache invalidation (not security), this is irrelevant
- Doesn't detect changes to chunking logic or embedding model — changing `CHUNK_SIZE` or `EMBEDDING_DIMENSIONS` won't trigger a rebuild. Manual delete of the hash file is required in those cases

---

## ADR-004: Informed query expansion

**Context**
Users often ask vague questions ("what's in this document?") that don't match any specific chunk. Query expansion — rewriting the question into multiple specific sub-questions before searching — helps, but blind expansion (no document context) produces generic sub-questions that still miss the mark.

**Decision**
Generate a one-paragraph document summary at index time using Haiku. Save it to disk. Pass the summary to Haiku when expanding queries, so sub-questions use actual document vocabulary.

**Alternatives considered**
- Blind expansion (no summary) — tried first, produced generic queries, discarded
- Pass the full document text to Haiku for expansion — accurate but expensive and hits token limits for long documents
- HyDE (Hypothetical Document Embeddings) — generate a hypothetical answer, embed it, search. More complex, better for some use cases, not implemented here

**Tradeoffs**
- Summary is generated once and cached — if the document changes significantly, the summary may go stale (but the MD5 check triggers a rebuild including a new summary)
- Summary truncated to 3000 characters — enough to capture topic and key themes, but may miss important detail from later sections of long documents

---

## ADR-005: Model routing — Haiku for preprocessing, Opus for answers

**Context**
The pipeline calls Claude multiple times per user question: query expansion, and the final answer. These tasks have very different quality requirements. The final answer must be accurate and well-reasoned. Query expansion just needs to produce reasonable search strings.

**Decision**
Use Haiku (fast, cheap) for summarization, query expansion, and eval generation. Use Opus (most capable) for the final answer the user actually sees.

**Alternatives considered**
- Use Opus for everything — higher quality but unnecessary cost for lightweight tasks
- Use Haiku for everything — cheaper but degrades answer quality where it matters
- Use Sonnet as a middle tier — reasonable, but the quality gap between Haiku and Opus is large enough that a two-tier routing is cleaner

**Tradeoffs**
- Adds model selection logic to maintain — if Haiku model IDs change, two places need updating
- Model routing is a standard production pattern; the complexity is justified at scale

---

## ADR-006: Keyword-based eval, not LLM-as-judge

**Context**
Testing retrieval quality requires some form of automated evaluation. Two main approaches: check whether expected keywords appear in the retrieved chunks, or ask a language model to judge whether the retrieved context is relevant.

**Decision**
Use keyword matching. Eval questions are generated by Haiku as question + keyword pairs. The pipeline retrieves chunks for each question and checks whether the expected keywords appear in the combined text.

**Alternatives considered**
- LLM-as-judge — ask Claude to rate whether the retrieved context answers the question. More nuanced, but adds cost and latency to every eval run, and introduces model bias (Claude may rate Claude-generated answers generously)
- Human evaluation — accurate but doesn't scale and requires domain expertise for niche documents
- Embedding similarity between expected and retrieved text — avoids exact keyword matching, but harder to interpret when it fails

**Tradeoffs**
- Keyword matching is brittle — synonyms, paraphrasing, and different vocabulary will cause false failures
- Fast and cheap: no LLM call needed for scoring, works for any document
- Good enough for catching retrieval gaps (a FAIL reliably means the right chunk wasn't retrieved); not good enough for measuring answer quality

---

## ADR-007: GitHub Actions CI — syntax check as foundation

**Context**
Without automated validation, broken code can be pushed to main undetected. For a learning repo with a growing number of scripts across multiple weeks, the risk of an old file breaking due to a dependency change increases over time.

**Decision**
Add a GitHub Actions workflow that runs on every push to main. It installs all dependencies and runs `py_compile` on each script. Catches syntax errors and import failures before they persist in the repo.

**Alternatives considered**
- Pre-commit hooks (local) — runs before every commit, but only on the committer's machine. Doesn't protect the repo from pushes from other environments
- Full integration tests — would require real AWS and Pinecone credentials in CI, adds cost and complexity. Right choice for Week 4 when there's a deployable API to test against
- Linting with flake8 or ruff — style enforcement is useful but lower priority than correctness for a learning project

**Tradeoffs**
- `py_compile` only catches syntax errors, not runtime failures, wrong logic, or API contract changes
- Running with real credentials in CI would provide much stronger guarantees — noted as a future step for Week 4
- Pinning action versions (`actions/checkout@v4.2.2`) prevents unexpected behaviour from upstream updates but requires manual version bumps
