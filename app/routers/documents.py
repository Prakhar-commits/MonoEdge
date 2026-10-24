import hashlib
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_embedder
from app.models import Chunk, Document
from app.schemas import DocumentListItem, DocumentOut
from app.services.embeddings import Embedder
from app.services.ingestion import IngestionError, count_chunks, extract_pages, ingest_document

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
}


def existing_response(db: Session, document: Document) -> DocumentOut:
    return DocumentOut(
        document_id=str(document.id),
        file_name=document.file_name,
        pages=document.page_count,
        chunks=count_chunks(db, document.id),
        duplicate=True,
    )


# Plain `def`, not `async def`: embedding is CPU-bound and blocking. FastAPI runs plain
# `def` routes in a threadpool, so one upload does not freeze the event loop for everyone else.
@router.post("/documents", status_code=201, response_model=DocumentOut)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
):
    settings = get_settings()
    file_name = os.path.basename(file.filename or "")
    extension = os.path.splitext(file_name)[1].lower()
    if extension not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, "Unsupported file type. Upload a .pdf, .txt or .md file.")
    if file.content_type not in ALLOWED_CONTENT_TYPES[extension]:
        raise HTTPException(415, f"Content type {file.content_type} does not match {extension}.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)  # read one extra byte to detect "too big"
    if len(data) > max_bytes:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    if extension == ".pdf" and not data.startswith(b"%PDF"):
        raise HTTPException(415, "File is not a valid PDF.")

    sha256 = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(Document).where(Document.sha256 == sha256))
    if existing:
        response.status_code = 200
        return existing_response(db, existing)

    try:
        pages = extract_pages(data, extension)
        if len(pages) > settings.max_pages:
            raise HTTPException(
                413, f"Document has {len(pages)} pages; the limit is {settings.max_pages}."
            )
        result = ingest_document(
            db,
            embedder,
            file_name=file_name,
            content_type=file.content_type,
            sha256=sha256,
            pages=pages,
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
        )
    except IngestionError as exc:
        raise HTTPException(422, str(exc))
    except IntegrityError:
        # Two identical uploads raced; the unique sha256 let only one win.
        db.rollback()
        response.status_code = 200
        return existing_response(db, db.scalar(select(Document).where(Document.sha256 == sha256)))

    return DocumentOut(
        document_id=str(result.document.id),
        file_name=file_name,
        pages=result.document.page_count,
        chunks=result.chunk_count,
        duplicate=False,
    )


@router.get("/documents", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
    ).all()
    return [
        DocumentListItem(
            document_id=str(doc.id),
            file_name=doc.file_name,
            content_type=doc.content_type,
            pages=doc.page_count,
            chunks=chunk_count,
            created_at=doc.created_at,
        )
        for doc, chunk_count in rows
    ]


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    db.delete(document)  # chunks go with it (ON DELETE CASCADE)
    db.commit()
