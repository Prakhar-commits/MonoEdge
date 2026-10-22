import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Ticket, WebhookEvent
from app.schemas import TicketData, TicketUpdateData

logger = logging.getLogger(__name__)


def create_ticket_from_event(db: Session, data: dict) -> None:
    ticket = TicketData.model_validate(data)
    # Upsert: if the ticket already exists (e.g. a retried event), overwrite its fields.
    stmt = pg_insert(Ticket).values(**ticket.model_dump(), source="webhook")
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticket_id"],
        set_={
            "title": stmt.excluded.title,
            "priority": stmt.excluded.priority,
            "description": stmt.excluded.description,
        },
    )
    db.execute(stmt)


def update_ticket_from_event(db: Session, data: dict) -> None:
    update = TicketUpdateData.model_validate(data)
    ticket = db.scalar(select(Ticket).where(Ticket.ticket_id == update.ticket_id))
    if ticket is None:
        raise ValueError(f"ticket {update.ticket_id} not found")
    # exclude_unset: only change fields the event actually sent.
    changes = update.model_dump(exclude_unset=True, exclude={"ticket_id"})
    for field, value in changes.items():
        setattr(ticket, field, value)


def process_event(event_id: str) -> None:
    """Background task. Uses its own session because the request's session is closed
    by the time this runs. Never raises: failures are recorded on the event row."""
    with SessionLocal() as db:
        try:
            event = db.get(WebhookEvent, event_id)
            if event.type == "ticket.created":
                create_ticket_from_event(db, event.payload["data"])
            elif event.type == "ticket.updated":
                update_ticket_from_event(db, event.payload["data"])
            # Any other type is stored and marked processed without side effects.
            event.status = "processed"
            event.error = None
            event.processed_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as exc:
            logger.exception("Failed to process webhook event %s", event_id)
            db.rollback()  # discard half-done ticket changes
            try:
                event = db.get(WebhookEvent, event_id)
                event.status = "failed"
                event.error = str(exc)[:2000]
                event.processed_at = datetime.now(timezone.utc)
                db.commit()
            except Exception:
                logger.exception("Could not record failure for event %s", event_id)
