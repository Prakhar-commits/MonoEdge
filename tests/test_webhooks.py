import threading
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text

from app.db import engine
from tests.conftest import make_event, post_signed, sign


def count(table: str) -> int:
    with engine.connect() as conn:
        return conn.scalar(text(f"SELECT count(*) FROM {table}"))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_webhook_invalid_signature_returns_401(client):
    body = make_event()
    response = client.post(
        "/webhooks/tickets", content=body, headers={"X-Signature": "sha256=" + "0" * 64}
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid signature"}
    assert count("webhook_events") == 0


def test_webhook_missing_signature_returns_401(client):
    response = client.post("/webhooks/tickets", content=make_event())
    assert response.status_code == 401


def test_webhook_valid_event_creates_ticket(client):
    response = post_signed(client, make_event())
    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "event_id": "evt_1"}

    tickets = client.get("/tickets").json()
    assert len(tickets) == 1
    assert tickets[0]["ticket_id"] == "TCK-1"
    assert tickets[0]["priority"] == "high"
    assert tickets[0]["source"] == "webhook"


def test_webhook_accepts_plain_hex_signature(client):
    body = make_event()
    response = client.post("/webhooks/tickets", content=body, headers={"X-Signature": sign(body)})
    assert response.status_code == 202


def test_webhook_duplicate_event_processed_once(client):
    body = make_event()
    assert post_signed(client, body).status_code == 202
    second = post_signed(client, body)
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert count("tickets") == 1
    assert count("webhook_events") == 1


def test_webhook_signature_checked_on_raw_bytes(client):
    compact = make_event()
    signature = sign(compact)
    headers = {"X-Signature": signature}

    assert client.post("/webhooks/tickets", content=compact, headers=headers).status_code == 202

    # Same JSON content, different spacing, old signature: must be rejected.
    spaced = compact.replace(b",", b", ").replace(b"evt_1", b"evt_2")
    assert client.post("/webhooks/tickets", content=spaced, headers=headers).status_code == 401


def test_webhook_invalid_body_returns_422(client):
    body = b'{"event_id": "evt_9"}'
    assert post_signed(client, body).status_code == 422


def test_webhook_invalid_priority_marks_event_failed(client):
    response = post_signed(client, make_event(priority="urgent"))
    assert response.status_code == 202  # accepted; the failure happens in the background
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, error FROM webhook_events")).one()
    assert row.status == "failed"
    assert "priority" in row.error
    assert count("tickets") == 0


def test_failed_event_is_retried_on_redelivery(client):
    bad = make_event(priority="urgent")
    assert post_signed(client, bad).status_code == 202
    # Same event_id redelivered (still invalid): a failed event is reprocessed, not ignored.
    assert post_signed(client, bad).status_code == 202


def test_ticket_updated_changes_only_sent_fields(client):
    post_signed(client, make_event(event_id="evt_1"))
    update = (
        b'{"event_id":"evt_2","type":"ticket.updated",'
        b'"data":{"ticket_id":"TCK-1","status":"closed"}}'
    )
    assert post_signed(client, update).status_code == 202
    ticket = client.get("/tickets").json()[0]
    assert ticket["status"] == "closed"
    assert ticket["title"] == "Login fails"


def test_unknown_event_type_is_stored_and_processed(client):
    body = b'{"event_id":"evt_3","type":"something.else","data":{}}'
    assert post_signed(client, body).status_code == 202
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT status FROM webhook_events")) == "processed"


def test_list_tickets_filters_by_priority(client):
    post_signed(client, make_event(event_id="e1", ticket_id="T1", priority="low"))
    post_signed(client, make_event(event_id="e2", ticket_id="T2", priority="high"))
    result = client.get("/tickets", params={"priority": "high"}).json()
    assert [t["ticket_id"] for t in result] == ["T2"]


def test_concurrent_duplicate_webhooks_create_exactly_one_ticket(client):
    """Eight identical requests at the same instant: the primary key lets only one win."""
    body = make_event(event_id="evt_race")
    start_together = threading.Barrier(8)

    def send(_):
        start_together.wait()
        return post_signed(client, body).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(send, range(8)))

    assert codes.count(202) == 1
    assert codes.count(200) == 7
    assert count("tickets") == 1
    assert count("webhook_events") == 1


def test_same_ticket_id_in_a_new_event_updates_the_ticket(client):
    post_signed(client, make_event(event_id="evt_a", title="Old title"))
    post_signed(client, make_event(event_id="evt_b", title="New title"))
    tickets = client.get("/tickets").json()
    assert [t["title"] for t in tickets] == ["New title"]
    assert count("webhook_events") == 2


def test_update_for_unknown_ticket_marks_event_failed(client):
    body = (
        b'{"event_id":"evt_x","type":"ticket.updated",'
        b'"data":{"ticket_id":"NOPE","status":"closed"}}'
    )
    assert post_signed(client, body).status_code == 202
    with engine.connect() as conn:
        row = conn.execute(text("SELECT status, error FROM webhook_events")).one()
    assert row.status == "failed"
    assert "NOPE" in row.error


def test_tickets_limit_is_capped_and_status_filter_works(client):
    for i in range(3):
        post_signed(client, make_event(event_id=f"e{i}", ticket_id=f"T{i}"))
    assert len(client.get("/tickets", params={"limit": 2}).json()) == 2
    assert len(client.get("/tickets", params={"limit": 100000}).json()) == 3
    assert client.get("/tickets", params={"status": "closed"}).json() == []
