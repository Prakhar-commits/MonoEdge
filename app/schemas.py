from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class WebhookEvent(BaseModel):
    event_id: str
    type: str
    data: dict


class TicketData(BaseModel):
    """Payload of a `ticket.created` event."""

    ticket_id: str
    title: str
    priority: Literal["low", "medium", "high"]
    description: str | None = None


class TicketUpdateData(BaseModel):
    """Payload of a `ticket.updated` event; only the fields present are changed."""

    ticket_id: str
    title: str | None = None
    priority: Literal["low", "medium", "high"] | None = None
    status: str | None = None
    description: str | None = None


class TicketOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticket_id: str
    title: str
    priority: str
    status: str
    description: str | None
    source: str
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    document_id: str
    file_name: str
    pages: int
    chunks: int
    duplicate: bool


class DocumentListItem(BaseModel):
    document_id: str
    file_name: str
    content_type: str
    pages: int
    chunks: int
    created_at: datetime


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    question: Question
    retrieval: Literal["vector", "keyword", "hybrid"] = "hybrid"
    rerank: bool = True
    hyde: bool = False
    top_k: int | None = Field(default=None, ge=1, le=20)  # None -> FINAL_TOP_K


class Source(BaseModel):
    file_name: str
    page: int
    score: float
    snippet: str


class AskDebug(BaseModel):
    retrieval: str
    rerank: bool
    hyde: bool
    candidates: int
    latency_ms: int


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]
    citations_missing: bool = False
    debug: AskDebug
