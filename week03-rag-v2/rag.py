import anthropic
import boto3
import json
import os
import hashlib
import time
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- Constants ---
# 512 dimensions: balance between accuracy and cost for Titan Embeddings V2.
# Supports 256, 512, 1024 — 512 is the standard starting point for enterprise RAG.
EMBEDDING_DIMENSIONS = 512

# Paragraph-aware chunking with 150-word limit.
# Word-count chunking was replaced because it splits sentences mid-paragraph,
# producing chunks with blurred meaning and weaker embeddings.
CHUNK_SIZE = 150

# Similarity score threshold for Pinecone results.
# Titan embeddings score conversational queries lower than keyword matches —
# 0.3 found empirically to balance precision vs. recall for this document type.
RELEVANCE_THRESHOLD = 0.3

# Return top 3 results per query to increase coverage across expanded queries.
TOP_K = 3

INDEX_NAME = "ai-learning-rag"
DOCUMENT_PATH = "week03-rag-v2/document.txt"
HASH_PATH = "week03-rag-v2/document.hash"
SUMMARY_PATH = "week03-rag-v2/document.summary"

# Model routing: Haiku for lightweight preprocessing tasks (query expansion, summarization),
# Opus for final answer generation where quality matters most.
# This reduces API cost without degrading the output the user actually sees.
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def get_document_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_document(filepath):
    if filepath.endswith(".pdf"):
        reader = PdfReader(filepath)
        return "\n\n".join(page.extract_text() for page in reader.pages)
    with open(filepath, "r") as f:
        return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_word_count = 0

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        if current_word_count + paragraph_word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        if paragraph_word_count > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            chunks.append(paragraph)
        else:
            current_chunk.append(paragraph)
            current_word_count += paragraph_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_embedding(bedrock_client, text):
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,
            "normalize": True  # required for cosine similarity via inner product
        })
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def summarize_document(anthropic_client, text):
    # Only first 3000 characters — enough to capture topic and key themes
    # without sending the full document for what is a lightweight task.
    message = anthropic_client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""Write a one-paragraph summary of this document.
Focus on the main topic, key themes, and type of information it contains.

Document:
{text[:3000]}"""
        }]
    )
    return message.content[0].text.strip()


def build_index(pinecone_index, chunks, bedrock_client):
    print(f"Embedding and uploading {len(chunks)} chunks to Pinecone...")

    # Clear existing vectors before rebuild — upsert by ID would also work
    # but delete_all is simpler when the full document has changed.
    stats = pinecone_index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        pinecone_index.delete(delete_all=True)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(bedrock_client, chunk)
        # Chunk text stored as metadata alongside the vector —
        # retrieved directly from Pinecone at query time, no separate file needed.
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })
        print(f"  Chunk {i + 1}/{len(chunks)} done")

    pinecone_index.upsert(vectors=vectors)
    print("Index uploaded to Pinecone.\n")


def search(pinecone_index, query_embedding):
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    return [
        match["metadata"]["text"]
        for match in results["matches"]
        if match["score"] >= RELEVANCE_THRESHOLD
    ]


def expand_query(anthropic_client, question, document_summary):
    # Informed query expansion: Haiku generates domain-specific sub-questions
    # using the document summary as context. Without the summary, expansion is
    # blind and produces generic queries that match poorly against document vocabulary.
    message = anthropic_client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""You are helping search a document. Here is a summary of the document:

{document_summary}

Generate 3 specific search queries to find relevant information for this question.
Return only the queries, one per line, no numbering, no explanation.

Question: {question}"""
        }]
    )
    lines = message.content[0].text.strip().split("\n")
    queries = [q.strip() for q in lines if q.strip()]
    # Original question always included — sub-questions supplement, not replace it.
    return [question] + queries


def ask(anthropic_client, question, context_chunks, conversation_history):
    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        user_message = f"""Use the following excerpts from a document to answer the question.
If the answer is not in the excerpts, say so.

Document excerpts:
{context}

Question: {question}"""
    else:
        # Graceful fallback to general knowledge when retrieval finds nothing relevant.
        # User is notified before the answer so they know the source.
        user_message = f"""Answer the following question using your general knowledge.

Question: {question}"""

    conversation_history.append({"role": "user", "content": user_message})

    message = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=conversation_history
    )

    reply = message.content[0].text
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def generate_eval_questions(anthropic_client, text, num_questions=5):
    # Haiku generates questions + expected keywords from the document automatically.
    # Replaces hardcoded test sets — works for any document without human involvement.
    # JSON format enforced so we can parse the response reliably.
    message = anthropic_client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Read this document and generate {num_questions} question and answer keyword pairs to test a retrieval system.

For each question, provide 2-3 keywords or short phrases that MUST appear in the retrieved text for the answer to be correct.

Return a JSON array only, no explanation. Format:
[
  {{"question": "...", "keywords": ["...", "..."]}},
  ...
]

Document:
{text}"""
        }]
    )
    response_text = message.content[0].text.strip()
    # LLMs sometimes wrap JSON in markdown code fences despite instructions —
    # this strips them defensively without breaking responses that don't have them.
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    return json.loads(response_text)


def run_eval(pinecone_index, bedrock_client, anthropic_client, text):
    print("\n--- Generating eval questions ---")
    eval_questions = generate_eval_questions(anthropic_client, text)
    print(f"Generated {len(eval_questions)} questions.\n")

    print("--- Running evaluation ---\n")
    passed = 0

    for item in eval_questions:
        question = item["question"]
        keywords = item["keywords"]

        query_embedding = get_embedding(bedrock_client, question)
        context_chunks = search(pinecone_index, query_embedding)
        combined = " ".join(context_chunks).lower()

        # Keyword presence check: tests retrieval quality without a Claude call.
        # A FAIL means the right chunk wasn't retrieved — signal to tune chunking or threshold.
        found = all(kw.lower() in combined for kw in keywords)
        status = "PASS" if found else "FAIL"
        if found:
            passed += 1

        print(f"[{status}] {question}")
        if not found:
            missing = [kw for kw in keywords if kw.lower() not in combined]
            print(f"       Missing keywords: {missing}")

    print(f"\nResult: {passed}/{len(eval_questions)} passed\n")


def get_or_create_index(pc):
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        # Serverless spec on AWS us-east-1 — only region available on Pinecone free tier.
        # For enterprise: use a region matching your data residency requirements.
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  Waiting for index to be ready...")
            time.sleep(2)
        print("Index ready.\n")

    return pc.Index(INDEX_NAME)


def main():
    bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    pinecone_index = get_or_create_index(pc)

    # MD5 hash check: rebuild index only when document content changes.
    # Prevents expensive re-embedding on every run for unchanged documents.
    current_hash = get_document_hash(DOCUMENT_PATH)
    needs_indexing = True

    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            saved_hash = f.read()
        if current_hash == saved_hash:
            print(f"Document unchanged. Using existing Pinecone index.\n")
            needs_indexing = False
        else:
            print("Document has changed. Rebuilding index...\n")

    if needs_indexing:
        text = load_document(DOCUMENT_PATH)
        chunks = chunk_text(text)
        build_index(pinecone_index, chunks, bedrock_client)
        with open(HASH_PATH, "w") as f:
            f.write(current_hash)
        print("Generating document summary...")
        document_summary = summarize_document(anthropic_client, text)
        with open(SUMMARY_PATH, "w") as f:
            f.write(document_summary)
        print(f"Summary: {document_summary}\n")
    else:
        with open(SUMMARY_PATH, "r") as f:
            document_summary = f.read()

    text = load_document(DOCUMENT_PATH)
    run_eval(pinecone_index, bedrock_client, anthropic_client, text)

    conversation_history = []
    print("Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ")

        if question == "quit":
            print("Goodbye!")
            break

        queries = expand_query(anthropic_client, question, document_summary)
        print(f"\nExpanded queries: {queries}\n")

        # Deduplicate chunks across all expanded queries using a set.
        # Same chunk may match multiple sub-questions — include it only once.
        all_chunks = []
        seen = set()
        for query in queries:
            query_embedding = get_embedding(bedrock_client, query)
            chunks = search(pinecone_index, query_embedding)
            for chunk in chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    all_chunks.append(chunk)

        context_chunks = all_chunks

        if not context_chunks:
            print("\nNote: nothing relevant found in the document — answering from general knowledge.\n")

        answer = ask(anthropic_client, question, context_chunks, conversation_history)
        print(f"\nClaude: {answer}\n")


if __name__ == "__main__":
    main()
