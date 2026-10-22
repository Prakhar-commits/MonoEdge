import hashlib
import hmac
import json
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

# The app reads its settings at import time, so set the environment BEFORE importing it.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://app:app@localhost:5433/docs_test"
)
TEST_SECRET = "test-secret"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["WEBHOOK_SECRET"] = TEST_SECRET


def _create_test_database_if_missing() -> None:
    url = make_url(TEST_DATABASE_URL)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
        )
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin.dispose()


_create_test_database_if_missing()

from fastapi.testclient import TestClient  # noqa: E402

from app.db import engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # Entering the context runs the app lifespan: creates the extension and tables once.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_tables(client):
    yield
    with engine.begin() as conn:
        conn.execute(
            text("TRUNCATE tickets, webhook_events, chunks, documents RESTART IDENTITY CASCADE")
        )


def sign(body: bytes, secret: str = TEST_SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def make_event(event_id="evt_1", type="ticket.created", **data_overrides) -> bytes:
    data = {"ticket_id": "TCK-1", "title": "Login fails", "priority": "high"}
    data.update(data_overrides)
    payload = {"event_id": event_id, "type": type, "data": data}
    return json.dumps(payload, separators=(",", ":")).encode()


def post_signed(client, body: bytes):
    return client.post(
        "/webhooks/tickets",
        content=body,
        headers={"X-Signature": f"sha256={sign(body)}", "Content-Type": "application/json"},
    )
