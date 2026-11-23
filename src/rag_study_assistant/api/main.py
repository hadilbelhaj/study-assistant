
"""FastAPI application for the RAG study assistant."""

import logging
from contextlib import asynccontextmanager
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from ollama import list as ollama_list

from rag_study_assistant.api.schemas import ChatRequest, ChatResponse
from rag_study_assistant.conversation_store import ConversationStore
from rag_study_assistant.qdrant_storage import get_qdrant_client
from rag_study_assistant.rag import rag_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start with one model request at a time.
inference_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ConversationStore()
    app.state.conversation_store = store

    logger.info("RAG Study Assistant API started.")

    try:
        yield
    finally:
        store.close()
        logger.info("RAG Study Assistant API stopped.")


app = FastAPI(
    title="RAG Study Assistant",
    description="AI-powered study assistant grounded in university course materials.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health/live")
def liveness():
    """Check whether the API process is running."""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness(request: Request):
    """Check whether the required services are reachable."""
    try:
        request.app.state.conversation_store.ping()

        qdrant_client = get_qdrant_client()
        try:
            qdrant_client.get_collections()
        finally:
            qdrant_client.close()

        ollama_list()

        return {"status": "ready"}

    except Exception:
        logger.exception("Readiness check failed.")
        raise HTTPException(
            status_code=503,
            detail="One or more assistant dependencies are unavailable.",
        )


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, request: Request):
    """Receive a question and return a grounded answer."""
    conversation_id = payload.conversation_id or uuid4()

    try:
        with inference_lock:
            result = rag_query(
                payload.message,
                conversation_id=str(conversation_id),
                conversation_store=request.app.state.conversation_store,
            )

        return ChatResponse(
            conversation_id=conversation_id,
            answer=result["answer"],
            sources=result.get("sources", []),
        )

    except Exception:
        logger.exception(
            "Chat request failed for conversation %s.",
            conversation_id,
        )
        raise HTTPException(
            status_code=500,
            detail="The assistant could not process your request.",
        )
