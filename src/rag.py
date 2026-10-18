
"""Main RAG entry point used by the CLI and notebooks."""
from src.conversation_store import ConversationStore
from src.generate import generate
from src.query_rewriter import rewrite_query
from src.retrieve import retrieve
from src.router import detect_smalltalk, REPLIES

def rag_query(
    query: str,
    metadata_filter: dict | None = None,
    *,
    conversation_id: str | None = None,
    conversation_store: ConversationStore | None = None,
) -> dict:
    """Answer a question using optional Redis-backed conversation history."""
    if (conversation_id is None) != (conversation_store is None):
        raise ValueError(
            "conversation_id and conversation_store must be provided together."
        )

    kind = detect_smalltalk(query)

    if kind:
        result = {"answer": REPLIES[kind], "sources": []}
    else:
        history = []

        if conversation_store is not None and conversation_id is not None:
            history = conversation_store.get_recent_messages(
                conversation_id, limit=10
            )

        has_previous_question = any(
            message.get("role") == "user"
            and detect_smalltalk(message.get("content", "")) is None
            for message in history
        )

        retrieval_query = (
            rewrite_query(query, history)
            if has_previous_question
            else query
        )

        chunks = retrieve(retrieval_query, metadata_filter=metadata_filter)
        print("REWRITTEN:", retrieval_query)
        print("CHUNKS FOUND:", len(chunks))
        for c in chunks:
            print("-----")
            print(c["text"])
        result = generate(retrieval_query, chunks)

    if conversation_store is not None and conversation_id is not None:
        conversation_store.add_message(conversation_id, "user", query)
        conversation_store.add_message(
            conversation_id, "assistant", result["answer"]
        )
    return result
