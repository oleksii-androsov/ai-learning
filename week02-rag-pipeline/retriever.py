import anthropic
import boto3
import json
import numpy as np
import faiss
import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIMENSIONS = 512
TOP_K = 2


def load_index():
    index = faiss.read_index("week02-rag-pipeline/index.faiss")
    with open("week02-rag-pipeline/chunks.json", "r") as f:
        chunks = json.load(f)
    return index, chunks


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


def search(index, chunks, query_embedding, top_k=TOP_K):
    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results


def ask(anthropic_client, question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""Use the following excerpts from a document to answer the question.
If the answer is not in the excerpts, say so.

Document excerpts:
{context}

Question: {question}"""

    message = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def main():
    bedrock_client = boto3.client("bedrock-runtime", region_name="eu-central-1")
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    index, chunks = load_index()
    print(f"Index loaded. {index.ntotal} vectors available.")
    print("Ask questions about the document. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ")

        if question == "quit":
            print("Goodbye!")
            break

        query_embedding = get_embedding(bedrock_client, question)
        context_chunks = search(index, chunks, query_embedding)
        answer = ask(anthropic_client, question, context_chunks)

        print(f"\nClaude: {answer}\n")


main()
