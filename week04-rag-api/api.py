import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import anthropic
import boto3
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.security.api_key import APIKeyHeader
from pinecone import Pinecone
from pydantic import BaseModel

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


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        })


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("rag-api")

API_KEY = os.environ["RAG_API_KEY"]
api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


clients = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading clients and checking index...")

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
        logger.info("Document changed or not indexed — rebuilding...")
        text = load_document(DOCUMENT_PATH)
        chunks = chunk_text(text)
        build_index(clients["index"], chunks, clients["bedrock"])
        with open(HASH_PATH, "w") as f:
            f.write(current_hash)
        clients["summary"] = summarize_document(clients["anthropic"], text)
        with open(SUMMARY_PATH, "w") as f:
            f.write(clients["summary"])
    else:
        logger.info("Document unchanged — using existing index.")
        with open(SUMMARY_PATH, "r") as f:
            clients["summary"] = f.read()

    logger.info("Ready.")
    yield


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000)
    logger.info(json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
    }))
    return response


class QuestionRequest(BaseModel):
    question: str
    history: list[dict] = []


class AnswerResponse(BaseModel):
    answer: str
    sources_found: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    with open(DOCUMENT_PATH, "wb") as f:
        f.write(contents)

    logger.info(f"Document uploaded: {file.filename}")

    text = load_document(DOCUMENT_PATH)
    chunks = chunk_text(text)
    build_index(clients["index"], chunks, clients["bedrock"])

    current_hash = get_document_hash(DOCUMENT_PATH)
    with open(HASH_PATH, "w") as f:
        f.write(current_hash)

    clients["summary"] = summarize_document(clients["anthropic"], text)
    with open(SUMMARY_PATH, "w") as f:
        f.write(clients["summary"])

    logger.info("Re-indexing complete.")
    return {"message": f"Document '{file.filename}' indexed successfully.", "chunks": len(chunks)}


@app.post("/ask", response_model=AnswerResponse, dependencies=[Depends(verify_api_key)])
def ask_question(request: QuestionRequest):
    queries = expand_query(clients["anthropic"], request.question, clients["summary"])
    context_chunks = retrieve_chunks(clients["index"], clients["bedrock"], queries)
    answer = ask(clients["anthropic"], request.question, context_chunks, request.history)
    return AnswerResponse(answer=answer, sources_found=bool(context_chunks))
