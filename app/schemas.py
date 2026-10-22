from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


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
