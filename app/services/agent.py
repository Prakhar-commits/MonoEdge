import json
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints, ValidationError
from sqlalchemy.orm import Session

from app.models import Ticket
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)

# Cheap routing: only questions that look like "raise a ticket ..." reach the tool-calling LLM,
# so normal questions pay for no extra LLM call.
TICKET_INTENT = re.compile(r"\b(raise|create|open|file|log|submit)\b.{0,40}\bticket\b", re.I)

SYSTEM_PROMPT = (
    "You help users raise support tickets. When the user wants a ticket raised, call "
    "create_ticket with a concise title and a priority inferred from their words "
    "(low, medium or high; use medium if unclear). If they do not want a ticket, do not call "
    "any tool."
)

CREATE_TICKET_TOOL = {
    "type": "function",
    "function": {
        "name": "create_ticket",
        "description": "Create a support ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short summary of the problem"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                "description": {"type": "string", "description": "Optional extra detail"},
            },
            "required": ["title", "priority"],
        },
    },
}

FIELD_HELP = {
    "title": "a title of 3 to 120 characters",
    "priority": "a priority: low, medium or high",
    "description": "a description of at most 2000 characters",
}


Description = Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]


class CreateTicketArgs(BaseModel):
    """Everything the model sends is untrusted input: validate it before saving anything."""

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=120)]
    priority: Literal["low", "medium", "high"]
    description: Description | None = None


@dataclass
class AgentResult:
    answer: str
    found: bool
    ticket: Ticket | None = None


def wants_ticket(question: str) -> bool:
    return bool(TICKET_INTENT.search(question))


def invalid_args_message(problems: list[str]) -> str:
    return "I didn't create a ticket. Please give me " + ", ".join(problems) + "."


def try_create_ticket(db: Session, llm: LLMClient, question: str) -> AgentResult | None:
    """Returns None when the model does not call the tool, so the caller falls back to RAG."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        response = llm.complete(messages, tools=[CREATE_TICKET_TOOL])
    except Exception:
        logger.exception("Agent LLM call failed; falling back to normal answering")
        return None

    call = next((c for c in response.tool_calls if c.name == "create_ticket"), None)
    if call is None:
        return None

    try:
        args = CreateTicketArgs.model_validate(json.loads(call.arguments))
    except json.JSONDecodeError:
        message = invalid_args_message(["a title and a priority (low, medium or high)"])
        return AgentResult(message, False)
    except ValidationError as exc:
        fields = []
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else "title"
            if FIELD_HELP.get(field, field) not in fields:
                fields.append(FIELD_HELP.get(field, field))
        return AgentResult(invalid_args_message(fields), False)

    ticket = Ticket(
        ticket_id=f"AI-{uuid.uuid4().hex[:6]}",
        title=args.title,
        priority=args.priority,
        description=args.description,
        status="open",
        source="agent",
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return AgentResult(
        f"Created ticket {ticket.ticket_id}: {ticket.title} ({ticket.priority})",
        True,
        ticket,
    )
