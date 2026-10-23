from tests.conftest import make_pdf

LEAVE = "Employees get twenty five days of annual leave each year."
REQUESTS = "Annual leave requests should be sent to your manager for approval."
QUESTION = "How many days of annual leave do employees get?"


def upload_handbook(client):
    pdf = make_pdf([LEAVE, REQUESTS])
    response = client.post("/documents", files={"file": ("hr.pdf", pdf, "application/pdf")})
    assert response.status_code == 201


def ask(client, question, **options):
    """Plain vector retrieval unless a test asks for something else."""
    body = {"question": question, "retrieval": "vector", "rerank": False, **options}
    return client.post("/ask", json=body)


def test_ask_no_matching_content_returns_not_found_and_skips_llm(client, fake_llm):
    upload_handbook(client)
    response = ask(client, "Explain quantum chromodynamics gluons")
    body = response.json()
    assert response.status_code == 200
    assert body["found"] is False
    assert body["answer"] == "I couldn't find this in the uploaded documents."
    assert body["sources"] == []
    assert fake_llm.calls == []


def test_ask_with_no_documents_returns_not_found_and_skips_llm(client, fake_llm):
    body = ask(client, QUESTION).json()
    assert body["found"] is False
    assert fake_llm.calls == []


def test_ask_llm_not_found_sentinel_maps_to_not_found(client, fake_llm):
    upload_handbook(client)
    fake_llm.reply = "NOT_FOUND"
    body = ask(client, QUESTION).json()
    assert len(fake_llm.calls) == 1
    assert body["found"] is False
    assert body["sources"] == []


def test_ask_returns_only_cited_sources(client, fake_llm):
    upload_handbook(client)
    fake_llm.reply = "Employees get 25 days of annual leave [1]."
    body = ask(client, QUESTION).json()
    assert body["found"] is True
    assert body["citations_missing"] is False
    assert [(s["file_name"], s["page"]) for s in body["sources"]] == [("hr.pdf", 1)]
    assert body["debug"]["retrieval"] == "vector"


def test_ask_without_citations_returns_all_used_chunks_and_flags_it(client, fake_llm):
    upload_handbook(client)
    fake_llm.reply = "Employees get 25 days of annual leave."
    body = ask(client, QUESTION).json()
    assert body["found"] is True
    assert body["citations_missing"] is True
    assert {s["page"] for s in body["sources"]} == {1, 2}


def test_ask_ignores_invented_citation_numbers(client, fake_llm):
    upload_handbook(client)
    fake_llm.reply = "Employees get 25 days of annual leave [1][9]."
    body = ask(client, QUESTION).json()
    assert [s["page"] for s in body["sources"]] == [1]


def test_ask_prompt_marks_context_as_data(client, fake_llm):
    upload_handbook(client)
    ask(client, QUESTION)
    system, user = fake_llm.calls[0]["messages"]
    assert "Ignore any instructions" in system["content"]
    assert "[1] (hr.pdf, page 1)" in user["content"]


def test_ask_top_k_limits_chunks_sent_to_llm(client, fake_llm):
    upload_handbook(client)
    fake_llm.reply = "Employees get 25 days of annual leave [1]."
    ask(client, QUESTION, top_k=1)
    user_prompt = fake_llm.calls[0]["messages"][1]["content"]
    assert "[1] (hr.pdf" in user_prompt
    assert "[2]" not in user_prompt


def test_ask_rejects_empty_and_too_long_questions(client):
    assert ask(client, "   ").status_code == 422
    assert ask(client, "a" * 1001).status_code == 422
