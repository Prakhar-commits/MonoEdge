from sqlalchemy import text

from app.db import engine
from tests.conftest import make_pdf

LEAVE = "Employees get twenty five days of annual leave each year."
APPROVAL = "Expenses above five thousand rupees need finance approval first."


def upload(client, name, data, content_type):
    return client.post("/documents", files={"file": (name, data, content_type)})


def test_upload_pdf_stores_chunks_with_page_numbers(client):
    response = upload(client, "handbook.pdf", make_pdf([LEAVE, APPROVAL]), "application/pdf")
    assert response.status_code == 201
    body = response.json()
    assert (body["pages"], body["chunks"], body["duplicate"]) == (2, 2, False)

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT page, content FROM chunks ORDER BY page")).all()
    assert [r.page for r in rows] == [1, 2]
    assert "annual leave" in rows[0].content
    assert "finance approval" in rows[1].content


def test_upload_text_file_is_one_page(client):
    response = upload(client, "notes.txt", LEAVE.encode(), "text/plain")
    assert response.status_code == 201
    assert response.json()["pages"] == 1


def test_upload_rejects_unsupported_type(client):
    response = upload(client, "malware.exe", b"MZ....", "application/octet-stream")
    assert response.status_code == 415


def test_upload_rejects_wrong_content_type_and_fake_pdf(client):
    assert upload(client, "a.pdf", b"%PDF-1.4", "text/plain").status_code == 415
    assert upload(client, "a.pdf", b"not a pdf", "application/pdf").status_code == 415


def test_upload_pdf_without_text_returns_422(client):
    response = upload(client, "blank.pdf", make_pdf([""]), "application/pdf")
    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_upload_duplicate_returns_existing(client):
    data = LEAVE.encode()
    first = upload(client, "notes.txt", data, "text/plain").json()
    second = upload(client, "renamed.txt", data, "text/plain")
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document_id"] == first["document_id"]
    assert len(client.get("/documents").json()) == 1


def test_list_and_delete_document(client):
    document_id = upload(client, "notes.txt", LEAVE.encode(), "text/plain").json()["document_id"]
    listing = client.get("/documents").json()
    assert listing[0]["chunks"] == 1
    assert client.delete(f"/documents/{document_id}").status_code == 204
    assert client.get("/documents").json() == []
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT count(*) FROM chunks")) == 0
    assert client.delete(f"/documents/{document_id}").status_code == 404


def test_upload_over_page_limit_returns_413(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "max_pages", 2)
    pdf = make_pdf([LEAVE, APPROVAL, LEAVE + "!"])
    response = upload(client, "long.pdf", pdf, "application/pdf")
    assert response.status_code == 413
    assert "3 pages" in response.json()["detail"]
    assert client.get("/documents").json() == []
