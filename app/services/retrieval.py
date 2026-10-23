import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk
from app.services.embeddings import Embedder
from app.services.llm import LLMClient
from app.services.reranker import Reranker

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the original RRF paper; damps the weight of top ranks

HYDE_PROMPT = (
    "Write a short passage (about 100 words) that would answer the question below, "
    "as it might appear in a company document. Output only the passage.\n\nQuestion: {question}"
)


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    scores: list[float]  # the score shown to the user (rerank score, else cosine, else RRF)
    candidates: int  # size of the candidate pool before reranking / trimming to top_k
    gate_passed: bool  # False -> answer "not found" without calling the LLM
    top_score: float | None = None  # the value the gate looked at (useful for calibration)
    hyde_used: bool = False
    cosine: dict = field(default_factory=dict)  # chunk id -> cosine similarity to the question


# ---------- vector search ----------


def vector_search(db: Session, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
    """Nearest chunks by cosine distance (`<=>`); similarity = 1 - distance."""
    distance = Chunk.embedding.cosine_distance(vector)
    rows = db.execute(select(Chunk, distance).order_by(distance).limit(limit)).all()
    return [(chunk, 1.0 - float(dist)) for chunk, dist in rows]


# ---------- keyword search ----------


def build_tsquery(question: str) -> str | None:
    """Turn a natural-language question into an OR query for Postgres full-text search.

    Why OR: plainto_tsquery / websearch_to_tsquery AND every word together, so a normal
    question ("what does error E-4012 mean?") almost never matches any single chunk.
    With OR, any chunk containing any term matches, and ts_rank_cd ranks the best ones first.
    """
    words = re.findall(r"[a-z0-9][a-z0-9\-_.]*", question.lower())
    terms = []
    for word in words:
        word = word.rstrip(".-_")
        if len(word) >= 2 and word not in terms:
            terms.append(word)
    if not terms:
        return None
    # Each term is wrapped in single quotes so Postgres treats it as a literal word.
    # The regex above already guarantees no quotes or operators can get in.
    return " | ".join(f"'{term}'" for term in terms)


def keyword_search(db: Session, question: str, limit: int) -> list[tuple[Chunk, float]]:
    tsquery_text = build_tsquery(question)
    if tsquery_text is None:
        return []
    query = func.to_tsquery("english", tsquery_text)
    rank = func.ts_rank_cd(Chunk.content_tsv, query)
    stmt = select(Chunk, rank).where(Chunk.content_tsv.op("@@")(query))
    rows = db.execute(stmt.order_by(rank.desc()).limit(limit)).all()
    return [(chunk, float(score)) for chunk, score in rows]


# ---------- fusion, rerank, HyDE ----------


def rrf(ranked_lists: list[list[uuid.UUID]], k: int = RRF_K) -> list[tuple[uuid.UUID, float]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank), rank starting at 1.

    Cosine similarity and ts_rank live on different scales, so adding them would be
    meaningless. RRF uses only the rank positions, so no normalisation is needed.
    """
    scores: dict[uuid.UUID, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def rerank_chunks(
    reranker: Reranker, question: str, chunks: list[Chunk]
) -> list[tuple[Chunk, float]]:
    scores = reranker.score(question, [chunk.content for chunk in chunks])
    return sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)


def hyde_passage(llm: LLMClient, question: str) -> str | None:
    """HyDE: ask the LLM to write a fake answer; its embedding often sits closer to real
    answer passages than the short question does. Failure is not fatal."""
    try:
        text = llm.complete([{"role": "user", "content": HYDE_PROMPT.format(question=question)}])
        return text.text.strip() or None
    except Exception:
        logger.exception("HyDE generation failed; continuing without it")
        return None


# ---------- pipeline ----------


def retrieve(
    db: Session,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLMClient,
    question: str,
    *,
    retrieval: str,
    rerank: bool,
    hyde: bool,
    top_k: int,
    candidates: int,
    similarity_floor: float,
    rerank_threshold: float,
) -> RetrievalResult:
    by_id: dict[uuid.UUID, Chunk] = {}
    ranked_lists: list[list[uuid.UUID]] = []
    cosine: dict[uuid.UUID, float] = {}

    def add_list(rows: list[tuple[Chunk, float]]) -> None:
        for chunk, _ in rows:
            by_id[chunk.id] = chunk
        ranked_lists.append([chunk.id for chunk, _ in rows])

    if retrieval in ("vector", "hybrid"):
        rows = vector_search(db, embedder.embed_query(question), candidates)
        cosine = {chunk.id: sim for chunk, sim in rows}
        add_list(rows)

    hyde_used = False
    if hyde and retrieval in ("vector", "hybrid"):
        passage = hyde_passage(llm, question)
        if passage:
            # An ADDITIONAL list: the original question vector is never replaced.
            # The passage is not fed to keyword search: invented terms would be noise.
            add_list(vector_search(db, embedder.embed_documents([passage])[0], candidates))
            hyde_used = True

    if retrieval in ("keyword", "hybrid"):
        add_list(keyword_search(db, question, candidates))

    if len(ranked_lists) > 1:
        fused = rrf(ranked_lists)
    else:
        fused = [(chunk_id, 0.0) for chunk_id in (ranked_lists[0] if ranked_lists else [])]
    fused_scores = dict(fused)
    pool = [by_id[chunk_id] for chunk_id, _ in fused[:candidates]]

    if rerank and pool:
        # Rerank and gate with the ORIGINAL question only. A fluent HyDE answer must not
        # be able to make an unanswerable question look answerable.
        ranked = rerank_chunks(reranker, question, pool)[:top_k]
        top_score = ranked[0][1]
        gate_passed = top_score >= rerank_threshold
    else:
        ranked = [(chunk, cosine.get(chunk.id, fused_scores.get(chunk.id, 0.0))) for chunk in pool]
        ranked = ranked[:top_k]
        if cosine:
            top_score = max(cosine.values())  # best vector similarity in the whole pool
            gate_passed = top_score >= similarity_floor
        else:  # keyword-only without rerank: no similarity to judge, any hit passes
            top_score = None
            gate_passed = bool(pool)

    return RetrievalResult(
        chunks=[chunk for chunk, _ in ranked],
        scores=[score for _, score in ranked],
        candidates=len(pool),
        gate_passed=gate_passed and bool(ranked),
        top_score=top_score,
        hyde_used=hyde_used,
        cosine=cosine,
    )
