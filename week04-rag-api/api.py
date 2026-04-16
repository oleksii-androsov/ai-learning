import os
import sys
from contextlib import asynccontextmanager

import anthropic
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI
from pinecone import Pinecone
from pydantic import BaseModel

# Tell Python where to find rag.py from week03.
# os.path.dirname(__file__) is the folder this file lives in (week04-rag-api).
# We go one level up (..) then into week03-rag-v2 to reach rag.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week03-rag-v2"))

from rag import (
    DOCUMENT_PATH,
    HASH_PATH,
    SUMMARY_PATH,
    build_index,
    chunk_text,
    expand_query,
    ask,
    get_document_hash,
    get_or_create_index,
    load_document,
    retrieve_chunks,
    summarize_document,
)

load_dotenv()

# Shared state: clients and index are created once at startup and reused for every request.
# Creating them per-request would be slow — each client initialization involves
# network connections and credential lookups.
clients = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — loading clients and checking index...")

    clients["bedrock"] = boto3.client("bedrock-runtime", region_name="eu-central-1")
    clients["anthropic"] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    clients["index"] = get_or_create_index(pc)

    current_hash = get_document_hash(DOCUMENT_PATH)
    needs_indexing = True

    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r") as f:
            if f.read() == current_hash:
                needs_indexing = False

    if needs_indexing:
        print("Document changed or not indexed — rebuilding...")
        text = load_document(DOCUMENT_PATH)
        chunks = chunk_text(text)
        build_index(clients["index"], chunks, clients["bedrock"])
        with open(HASH_PATH, "w") as f:
            f.write(current_hash)
        clients["summary"] = summarize_document(clients["anthropic"], text)
        with open(SUMMARY_PATH, "w") as f:
            f.write(clients["summary"])
    else:
        print("Document unchanged — using existing index.")
        with open(SUMMARY_PATH, "r") as f:
            clients["summary"] = f.read()

    print("Ready.\n")
    yield
    # Nothing to clean up on shutdown


app = FastAPI(lifespan=lifespan)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources_found: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    queries = expand_query(clients["anthropic"], request.question, clients["summary"])
    context_chunks = retrieve_chunks(clients["index"], clients["bedrock"], queries)
    answer = ask(clients["anthropic"], request.question, context_chunks, [])
    return AnswerResponse(answer=answer, sources_found=bool(context_chunks))
