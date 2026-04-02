import anthropic
import boto3
import json
import numpy as np
import faiss
import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMENSIONS = 512
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 2
INDEX_PATH = "week02-rag-pipeline/index.faiss"
CHUNKS_PATH = "week02-rag-pipeline/chunks.json"
DOCUMENT_PATH = "week02-rag-pipeline/document.txt"
HASH_PATH = "week02-rag-pipeline/document.hash"


def get_document_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_document(filepath):
    with open(filepath, "r") as f:
        return f.read()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
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


def build_index(chunks, bedrock_client):
    print(f"Building index from {len(chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(bedrock_client, chunk)
        embeddings.append(embedding)
        print(f"  Chunk {i + 1}/{len(chunks)} done")
    vectors = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(vectors)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    with open(HASH_PATH, "w") as f:
        f.write(get_document_hash(DOCUMENT_PATH))
    print("Index saved to disk.\n")
    return index, chunks


def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)
    print(f"Index loaded from disk. {index.ntotal} vectors available.\n")
    return index, chunks


def search(index, chunks, query_embedding):
    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, TOP_K)
    return [chunks[i] for i in indices[0]]


def ask(anthropic_client, question, context_chunks, conversation_history):
    context = "\n\n---\n\n".join(context_chunks)
    user_message = f"""Use the following excerpts from a document to answer the question.
If the answer is not in the excerpts, say so.

Document excerpts:
{context}

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


def main():
    bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    current_hash = get_document_hash(DOCUMENT_PATH)
    index_exists = os.path.exists(INDEX_PATH) and os.path.exists(HASH_PATH)

    if index_exists:
        with open(HASH_PATH, "r") as f:
            saved_hash = f.read()

    if index_exists and current_hash == saved_hash:
        index, chunks = load_index()
    else:
        if index_exists:
            print("Document has changed. Rebuilding index...\n")
        text = load_document(DOCUMENT_PATH)
        chunks = chunk_text(text)
        index, chunks = build_index(chunks, bedrock_client)

    conversation_history = []

    print("Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ")

        if question == "quit":
            print("Goodbye!")
            break

        query_embedding = get_embedding(bedrock_client, question)
        context_chunks = search(index, chunks, query_embedding)
        answer = ask(anthropic_client, question, context_chunks, conversation_history)

        print(f"\nClaude: {answer}\n")


main()
