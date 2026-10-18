
"""Smoke test for Redis-backed conversation history.

Run from the project root:
    python -m scripts.check_conversation_store
"""

from src.conversation_store import ConversationStore


def main() -> None:
    store = ConversationStore(max_messages=5, ttl_seconds=60)
    conversation_a = "redis-smoke-test-a"
    conversation_b = "redis-smoke-test-b"

    try:
        print("Redis reachable:", store.ping())

        # Start with clean test conversations.
        store.clear_conversation(conversation_a)
        store.clear_conversation(conversation_b)

        # Add six messages to a history limited to five.
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            store.add_message(conversation_a, role, f"message-{i}")

        # A separate conversation must remain isolated.
        store.add_message(conversation_b, "user", "private-message")

        history = store.get_recent_messages(conversation_a, limit=10)
        other_history = store.get_recent_messages(conversation_b, limit=10)

        assert len(history) == 5, "The message limit was not enforced."
        assert [m["content"] for m in history] == [
            "message-1", "message-2", "message-3", "message-4", "message-5"
        ], "Messages are missing or out of order."

        assert len(other_history) == 1
        assert other_history[0]["content"] == "private-message"
        assert all(m["content"] != "private-message" for m in history)

        print("Message ordering: OK")
        print("Maximum history size: OK")
        print("Conversation isolation: OK")
        print("Stored message example:", history[-1])

        store.clear_conversation(conversation_a)
        store.clear_conversation(conversation_b)
        print("Conversation cleanup: OK")

    finally:
        store.close()


if __name__ == "__main__":
    main()
