import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Ticket, WebhookEvent
from app.schemas import TicketOut
from app.schemas import WebhookEvent as WebhookEventIn
from app.security import verify_signature
from app.services.tickets import process_event

router = APIRouter()


@router.post("/webhooks/tickets", status_code=202)
def receive_ticket_webhook(
    response: Response,
    background_tasks: BackgroundTasks,
    raw_body: bytes = Depends(verify_signature),  # 401 happens before anything else
    db: Session = Depends(get_db),
):
    try:
        event = WebhookEventIn.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json(include_url=False)))

    # Claim the event atomically. event_id is the primary key, so even if two identical
    # requests arrive at the same moment, the database lets exactly one INSERT succeed;
    # the other gets no row back. No "check then insert" race.
    claim = (
        pg_insert(WebhookEvent)
        .values(
            event_id=event.event_id,
            type=event.type,
            payload=event.model_dump(),
            status="received",
        )
        .on_conflict_do_nothing(index_elements=["event_id"])
        .returning(WebhookEvent.event_id)
    )
    claimed = db.execute(claim).first()
    if claimed is None:
        # Already seen. A failed event deserves a retry: reset it atomically (only one
        # concurrent request can flip failed -> received) and process it again.
        reset = (
            update(WebhookEvent)
            .where(WebhookEvent.event_id == event.event_id, WebhookEvent.status == "failed")
            .values(status="received", error=None, processed_at=None)
            .returning(WebhookEvent.event_id)
        )
        claimed = db.execute(reset).first()
    db.commit()

    if claimed is None:
        response.status_code = 200
        return {"status": "duplicate", "event_id": event.event_id}

    # Respond fast; the real work happens after the response is sent.
    background_tasks.add_task(process_event, event.event_id)
    return {"status": "accepted", "event_id": event.event_id}


@router.get("/tickets", response_model=list[TicketOut])
def list_tickets(
    priority: str | None = None,
    status: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 200))
    query = select(Ticket).order_by(Ticket.created_at.desc(), Ticket.id.desc()).limit(limit)
    if priority:
        query = query.where(Ticket.priority == priority)
    if status:
        query = query.where(Ticket.status == status)
    return db.scalars(query).all()
