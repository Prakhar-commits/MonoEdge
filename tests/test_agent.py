import json

from app.services.llm import ToolCall

RAISE = "Please raise a ticket: login fails for every user since this morning"


def tool_call(**arguments):
    return [ToolCall(name="create_ticket", arguments=json.dumps(arguments))]


def test_agent_creates_ticket_from_valid_tool_call(client, fake_llm):
    fake_llm.tool_calls = tool_call(title="Login fails", priority="high")
    body = client.post("/ask", json={"question": RAISE}).json()

    assert body["found"] is True
    assert body["sources"] == []
    assert body["action"]["type"] == "ticket_created"
    ticket = body["action"]["ticket"]
    assert ticket["ticket_id"].startswith("AI-")
    assert body["answer"] == f"Created ticket {ticket['ticket_id']}: Login fails (high)"

    saved = client.get("/tickets").json()
    assert [(t["ticket_id"], t["source"], t["priority"]) for t in saved] == [
        (ticket["ticket_id"], "agent", "high")
    ]


def test_agent_rejects_invalid_priority_without_saving(client, fake_llm):
    fake_llm.tool_calls = tool_call(title="Login fails", priority="urgent")
    body = client.post("/ask", json={"question": RAISE}).json()

    assert body["found"] is False
    assert body["action"] is None
    assert "priority" in body["answer"]
    assert client.get("/tickets").json() == []


def test_agent_rejects_too_short_title_and_bad_json(client, fake_llm):
    fake_llm.tool_calls = tool_call(title="x", priority="low")
    body = client.post("/ask", json={"question": RAISE}).json()
    assert "title" in body["answer"]

    fake_llm.tool_calls = [ToolCall(name="create_ticket", arguments="{not json")]
    body = client.post("/ask", json={"question": RAISE}).json()
    assert body["found"] is False
    assert client.get("/tickets").json() == []


def test_normal_question_does_not_trigger_agent(client, fake_llm):
    client.post("/ask", json={"question": "How many days of annual leave do employees get?"})
    assert all(call["tools"] is None for call in fake_llm.calls)
    assert client.get("/tickets").json() == []


def test_agent_falls_back_to_normal_answering_when_no_tool_is_called(client, fake_llm):
    fake_llm.tool_calls = []  # the model chooses not to call the tool
    body = client.post("/ask", json={"question": "How do I raise a ticket for IT?"}).json()
    assert body["found"] is False  # no documents uploaded, so the normal not-found answer
    assert body["action"] is None
    assert body["debug"]["retrieval"] == "hybrid"
