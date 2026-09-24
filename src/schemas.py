from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: UUID
    document_id: str

    text: str = Field(min_length=1)

    course: str
    study_year: str
    semester: Literal["S1", "S2"]

    lecture: str
    doc_type: Literal["lecture", "exercise", "exam"]

    language: Literal["fr", "en"]

    source_file: str
    page: int
    chunk_index: int