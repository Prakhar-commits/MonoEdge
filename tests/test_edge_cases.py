import pytest

from app.config import get_settings
from app.services.answer import NOT_FOUND_TEXT, parse_llm_answer
from app.services.chunking import chunk_pages
from app.services.retrieval import build_tsquery, rrf
from tests.conftest import make_pdf

LEAVE = "Employees get twenty five days of annual leave each year."


def upload(client, name, data, content_type):
    return client.post("/documents", files={"file": (name, data, content_type)})


# ---------- uploads ----------


def test_upload_larger_than_limit_returns_413_and_stores_nothing(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 0)  # any non-empty file is too big
    response = upload(client, "notes.txt", LEAVE.encode(), "text/plain")
    assert response.status_code == 413
    assert client.get("/documents").json() == []


def test_upload_exactly_at_the_limit_is_accepted(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_upload_mb", 1)
    data = ((LEAVE + " ").encode() * 20000)[: 1024 * 1024]  # more than enough, cut to 1 MiB
    assert len(data) == 1024 * 1024
    assert upload(client, "big.txt", data, "text/plain").status_code == 201


def test_corrupt_pdf_returns_422_not_500(client):
    response = upload(client, "broken.pdf", b"%PDF-1.4 this is not really a pdf", "application/pdf")
    assert response.status_code == 422


def test_markdown_upload_and_filename_is_sanitised(client):
    response = upload(client, "../../etc/notes.md", LEAVE.encode(), "text/markdown")
    assert response.status_code == 201
    assert response.json()["file_name"] == "notes.md"  # directory parts are stripped


def test_failed_ingestion_leaves_no_partial_document(client):
    upload(client, "blank.pdf", make_pdf([""]), "application/pdf")
    assert client.get("/documents").json() == []


def test_delete_with_invalid_id_returns_422(client):
    assert client.delete("/documents/not-a-uuid").status_code == 422


# ---------- /ask ----------


def test_llm_failure_returns_502_with_a_message(client, fake_llm, monkeypatch):
    upload(client, "hr.txt", LEAVE.encode(), "text/plain")

    def broken(messages, tools=None):
        raise RuntimeError("provider down")

    monkeypatch.setattr(fake_llm, "complete", broken)
    response = client.post(
        "/ask",
        json={"question": "How many days of annual leave do employees get?",
              "retrieval": "vector", "rerank": False},
    )
    assert response.status_code == 502
    assert "LLM request failed" in response.json()["detail"]


def test_keyword_only_retrieval_works_without_embedding_a_match(client, fake_llm):
    text = b"Error code E-4012 means the API token has expired and must be renewed."
    upload(client, "faq.txt", text, "text/plain")
    fake_llm.reply = "The token expired [1]."
    body = client.post(
        "/ask", json={"question": "What is E-4012?", "retrieval": "keyword", "rerank": False}
    ).json()
    assert body["found"] is True
    assert body["debug"]["retrieval"] == "keyword"


def test_top_k_validation(client):
    for bad in (0, -1, 21):
        response = client.post("/ask", json={"question": "hello there", "top_k": bad})
        assert response.status_code == 422


def test_unknown_retrieval_mode_is_rejected(client):
    response = client.post("/ask", json={"question": "hello there", "retrieval": "magic"})
    assert response.status_code == 422


# ---------- answer parsing ----------


class StubChunk:
    def __init__(self, file_name, page, content="text"):
        self.file_name, self.page, self.content = file_name, page, content


@pytest.mark.parametrize("raw", ["NOT_FOUND", "not_found", "  NOT_FOUND.  ", "NOT_FOUND - no info"])
def test_not_found_sentinel_variants(raw):
    parsed = parse_llm_answer(raw, [StubChunk("a.pdf", 1)])
    assert parsed.found is False
    assert parsed.text == NOT_FOUND_TEXT
    assert parsed.used == []


def test_citations_are_deduplicated_by_file_and_page():
    chunks = [StubChunk("a.pdf", 1), StubChunk("a.pdf", 1), StubChunk("b.pdf", 2)]
    parsed = parse_llm_answer("Facts [1] and [2] and [3].", chunks)
    assert [(c.file_name, c.page) for c in parsed.used] == [("a.pdf", 1), ("b.pdf", 2)]
    assert parsed.citations_missing is False


def test_zero_and_out_of_range_citations_are_ignored():
    parsed = parse_llm_answer("Nothing real [0] [5].", [StubChunk("a.pdf", 1)])
    assert parsed.citations_missing is True
    assert [(c.file_name, c.page) for c in parsed.used] == [("a.pdf", 1)]


# ---------- pure-function edge cases ----------


def test_empty_inputs_do_not_crash():
    assert chunk_pages([], 2000, 300) == []
    assert chunk_pages([(1, "")], 2000, 300) == []
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert build_tsquery("") is None


def test_rrf_with_a_single_list_keeps_its_order():
    import uuid

    ids = [uuid.uuid4() for _ in range(3)]
    assert [chunk_id for chunk_id, _ in rrf([ids])] == ids
