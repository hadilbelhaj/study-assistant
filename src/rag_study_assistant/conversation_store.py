
"""Short-term conversation history backed by Redis."""

import json
import os
from datetime import datetime, timezone
from typing import Literal, TypedDict

import redis


Role = Literal["user", "assistant"]


class ConversationMessage(TypedDict):
    role: Role
    content: str
    created_at: str


class ConversationStore:
    def __init__(self, redis_url: str | None = None, max_messages: int = 20, ttl_seconds: int = 86400):
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1.")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be at least 1.")

        self.max_messages = max_messages
        self.ttl_seconds = ttl_seconds

        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = redis.Redis.from_url(url, decode_responses=True)

    def ping(self) -> bool:
        """Check whether Redis is reachable."""
        return bool(self._client.ping())

    def _key(self, conversation_id: str) -> str:
        conversation_id = conversation_id.strip()
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty.")

        return f"study-assistant:conversation:{conversation_id}:messages"

    def add_message(self, conversation_id: str, role: Role, content: str) -> None:
        """Append a message, enforce the history limit, and refresh its TTL."""
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'.")
        if not content.strip():
            raise ValueError("content cannot be empty.")

        message: ConversationMessage = {
            "role": role,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        key = self._key(conversation_id)
        serialized = json.dumps(message, ensure_ascii=False)

        # Execute all three operations together.
        with self._client.pipeline(transaction=True) as pipeline:
            pipeline.rpush(key, serialized)
            pipeline.ltrim(key, -self.max_messages, -1)
            pipeline.expire(key, self.ttl_seconds)
            pipeline.execute()

    def get_recent_messages(self, conversation_id: str, limit: int = 10) -> list[ConversationMessage]:
        """Return the most recent messages in chronological order."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")

        key = self._key(conversation_id)
        serialized_messages = self._client.lrange(key, -limit, -1)

        if serialized_messages:
            self._client.expire(key, self.ttl_seconds)

        return [json.loads(item) for item in serialized_messages]

    def clear_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation's history. Return whether a key existed."""
        return bool(self._client.delete(self._key(conversation_id)))

    def close(self) -> None:
        """Close the Redis client connection pool."""
        self._client.close()
