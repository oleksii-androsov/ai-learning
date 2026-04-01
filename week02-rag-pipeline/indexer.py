import boto3
import json
import numpy as np
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMENSIONS = 512
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


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
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = []

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(bedrock_client, chunk)
        embeddings.append(embedding)
        print(f"  Chunk {i + 1}/{len(chunks)} done")

    vectors = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
    index.add(vectors)

    return index


def main():
    bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")

    text = load_document("week02-rag-pipeline/document.txt")
    chunks = chunk_text(text)

    print(f"Document split into {len(chunks)} chunks")
    print(f"First chunk preview:\n{chunks[0][:200]}\n")

    index = build_index(chunks, bedrock_client)

    faiss.write_index(index, "week02-rag-pipeline/index.faiss")

    with open("week02-rag-pipeline/chunks.json", "w") as f:
        json.dump(chunks, f)

    print(f"\nIndex saved. {index.ntotal} vectors stored.")


main()
