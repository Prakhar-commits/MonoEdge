import uuid

import pytest

from app.config import get_settings
from app.db import SessionLocal
from app.services.retrieval import build_tsquery, keyword_search, retrieve, rrf, vector_search
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker

ERROR_DOC = (
    "Error code E-4012 means the API token has expired and must be regenerated "
    "from the account settings page. "
)
FILLER = (
    "Separately the quarterly roadmap mentions telemetry dashboards observability "
    "capacity forecasting procurement onboarding localisation accessibility "
    "benchmarking compliance sustainability partnerships."
)
DECOY_1 = "How to reset your account password using the self service portal."
DECOY_2 = "Password policy requires a reset every ninety days for all accounts."


def upload_text(client, name, text):
    files = {"file": (name, text.encode(), "text/plain")}
    assert client.post("/documents", files=files).status_code == 201


def run_retrieve(db, question, *, llm=None, reranker=None, **overrides):
    options = {
        "retrieval": "hybrid",
        "rerank": False,
        "hyde": False,
        "top_k": 5,
        "candidates": 20,
        "similarity_floor": 0.0,
        "rerank_threshold": 0.0,
    }
    options.update(overrides)
    return retrieve(
        db, FakeEmbedder(), reranker or FakeReranker(), llm or FakeLLM(), question, **options
    )


def test_rrf_combines_ranks_correctly():
    a, b, c, d = (uuid.uuid4() for _ in range(4))
    fused = rrf([[a, b, c], [b, c, d]], k=60)
    scores = dict(fused)
    assert scores[b] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[c] == pytest.approx(1 / 63 + 1 / 62)
    assert scores[a] == pytest.approx(1 / 61)
    assert scores[d] == pytest.approx(1 / 63)
    assert [chunk_id for chunk_id, _ in fused] == [b, c, a, d]


def test_keyword_query_builder_uses_or_and_escapes():
    query = build_tsquery("What does error E-4012 mean? x'); DROP TABLE chunks;--")
    assert " | " in query
    assert "'e-4012'" in query
    assert "'what'" in query and "'drop'" in query
    assert ";" not in query and ")" not in query
    assert "'x'" not in query  # single-character terms are dropped
    assert build_tsquery("? ! a") is None


def test_keyword_search_finds_exact_identifier(client):
    upload_text(client, "errors.txt", ERROR_DOC)
    upload_text(client, "other.txt", "The cafeteria menu changes every single week of the year.")
    with SessionLocal() as db:
        rows = keyword_search(db, "what does E-4012 mean", 5)
    assert [chunk.file_name for chunk, _ in rows] == ["errors.txt"]


def test_hybrid_returns_chunk_found_only_by_keyword(client):
    upload_text(client, "errors.txt", ERROR_DOC + FILLER)
    upload_text(client, "d1.txt", DECOY_1)
    upload_text(client, "d2.txt", DECOY_2)
    question = "password reset error E-4012"
    with SessionLocal() as db:
        # With room for only 2 candidates, vector search fills up with the two password decoys.
        nearest = vector_search(db, FakeEmbedder().vector(question), 2)
        assert "errors.txt" not in [chunk.file_name for chunk, _ in nearest]

        result = run_retrieve(db, question, candidates=2)
        assert "errors.txt" in [chunk.file_name for chunk in result.chunks]


def test_rerank_reorders_candidates_by_cross_encoder_score(client):
    upload_text(client, "d1.txt", DECOY_1)
    upload_text(client, "d2.txt", DECOY_2)
    with SessionLocal() as db:
        result = run_retrieve(db, "ninety days password reset", rerank=True, retrieval="vector")
    assert result.chunks[0].file_name == "d2.txt"
    assert result.scores == sorted(result.scores, reverse=True)


def test_rerank_gate_returns_not_found_below_threshold(client, fake_llm, monkeypatch):
    upload_text(client, "hr.txt", "Employees get twenty five days of annual leave each year.")
    monkeypatch.setattr(get_settings(), "rerank_score_threshold", 0.9)
    body = client.post("/ask", json={"question": "How many days of annual leave do employees get?"})
    assert body.json()["found"] is False
    assert fake_llm.calls == []


def test_hyde_reranks_with_original_question(client, fake_llm, fake_reranker):
    upload_text(client, "hr.txt", "Employees get twenty five days of annual leave each year.")
    fake_llm.reply = "Staff receive twenty five days of annual leave every year [1]."
    question = "How many days of annual leave do employees get?"
    body = client.post("/ask", json={"question": question, "hyde": True}).json()
    assert body["debug"]["hyde"] is True
    assert len(fake_llm.calls) == 2  # one for HyDE, one for the answer
    assert fake_reranker.queries == [question]  # never the hypothetical passage


def test_hyde_failure_does_not_break_the_request(client, fake_llm, monkeypatch):
    upload_text(client, "hr.txt", "Employees get twenty five days of annual leave each year.")
    real_complete = fake_llm.complete
    calls = {"n": 0}

    def flaky(messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("HyDE provider down")
        return real_complete(messages, tools)

    monkeypatch.setattr(fake_llm, "complete", flaky)
    fake_llm.reply = "Employees get twenty five days of leave [1]."
    body = client.post(
        "/ask", json={"question": "How many days of annual leave do employees get?", "hyde": True}
    ).json()
    assert body["found"] is True
    assert body["debug"]["hyde"] is False
