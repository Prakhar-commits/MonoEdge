import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PyPdfError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.services.chunking import chunk_pages
from app.services.embeddings import Embedder

EMBED_BATCH_SIZE = 32


class IngestionError(Exception):
    """Raised for files we cannot read; the router turns it into a 422."""


@dataclass
class IngestResult:
    document: Document
    chunk_count: int
    duplicate: bool


def extract_pages(data: bytes, extension: str) -> list[tuple[int, str]]:
    """Return [(page_number, text)] with 1-based page numbers."""
    if extension == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except PyPdfError as exc:
            raise IngestionError(f"Could not read PDF: {exc}") from exc
    # Text files are treated as a single page 1.
    return [(1, data.decode("utf-8", errors="replace"))]


def embed_in_batches(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(embedder.embed_documents(texts[start : start + EMBED_BATCH_SIZE]))
    return vectors


def count_chunks(db: Session, document_id) -> int:
    query = select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    return db.scalar(query)


def ingest_document(
    db: Session,
    embedder: Embedder,
    *,
    file_name: str,
    content_type: str,
    sha256: str,
    pages: list[tuple[int, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise IngestionError("No extractable text found (no text layer; OCR not supported).")

    # Embed before touching the database so a slow model never holds a transaction open.
    vectors = embed_in_batches(embedder, [c.text for c in chunks])

    # Document and chunks are saved in one transaction: either all of it or none of it.
    document = Document(
        file_name=file_name, content_type=content_type, sha256=sha256, page_count=len(pages)
    )
    db.add(document)
    db.flush()  # assigns document.id for the foreign key below
    db.add_all(
        Chunk(
            document_id=document.id,
            file_name=file_name,
            page=chunk.page,
            chunk_index=chunk.index,
            content=chunk.text,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, vectors)
    )
    db.commit()
    return IngestResult(document=document, chunk_count=len(chunks), duplicate=False)
