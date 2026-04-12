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

EMBEDDING_DIMENSIONS = 512
CHUNK_SIZE = 150
RELEVANCE_THRESHOLD = 0.3
TOP_K = 3
INDEX_NAME = "ai-learning-rag"
DOCUMENT_PATH = "week03-rag-v2/document.txt"
HASH_PATH = "week03-rag-v2/document.hash"


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
            "normalize": True
        })
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def build_index(pinecone_index, chunks, bedrock_client):
    print(f"Embedding and uploading {len(chunks)} chunks to Pinecone...")

    stats = pinecone_index.describe_index_stats()
    if stats["total_vector_count"] > 0:
        pinecone_index.delete(delete_all=True)

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(bedrock_client, chunk)
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


def ask(anthropic_client, question, context_chunks, conversation_history):
    if context_chunks:
        context = "\n\n---\n\n".join(context_chunks)
        user_message = f"""Use the following excerpts from a document to answer the question.
If the answer is not in the excerpts, say so.

Document excerpts:
{context}

Question: {question}"""
    else:
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


def get_or_create_index(pc):
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
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

    conversation_history = []
    print("Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ")

        if question == "quit":
            print("Goodbye!")
            break

        query_embedding = get_embedding(bedrock_client, question)
        context_chunks = search(pinecone_index, query_embedding)

        if not context_chunks:
            print("\nNote: nothing relevant found in the document — answering from general knowledge.\n")

        answer = ask(anthropic_client, question, context_chunks, conversation_history)
        print(f"\nClaude: {answer}\n")


main()
