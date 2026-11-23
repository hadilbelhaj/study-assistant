
"""Request and response models for the chat API."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None


class SourceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    chunk_id: str | None = None
    course: str | None = None
    lecture: str | None = None
    section: str | None = None
    source_file: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
