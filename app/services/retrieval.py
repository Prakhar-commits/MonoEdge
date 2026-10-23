from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.services.embeddings import Embedder


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    scores: list[float]  # cosine similarity, same order as `chunks`
    candidates: int  # how many candidates the search returned before trimming to top_k


def vector_search(
    db: Session, embedder: Embedder, question: str, limit: int
) -> list[tuple[Chunk, float]]:
    """Nearest chunks by cosine distance (`<=>`); similarity = 1 - distance."""
    query_vector = embedder.embed_query(question)
    distance = Chunk.embedding.cosine_distance(query_vector)
    rows = db.execute(select(Chunk, distance).order_by(distance).limit(limit)).all()
    return [(chunk, 1.0 - float(dist)) for chunk, dist in rows]


def retrieve(
    db: Session, embedder: Embedder, question: str, *, candidates: int, top_k: int
) -> RetrievalResult:
    found = vector_search(db, embedder, question, candidates)
    top = found[:top_k]
    return RetrievalResult(
        chunks=[chunk for chunk, _ in top],
        scores=[score for _, score in top],
        candidates=len(found),
    )
