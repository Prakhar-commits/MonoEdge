import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_embedder, get_llm
from app.schemas import AskDebug, AskRequest, AskResponse, Source
from app.services.answer import NOT_FOUND_TEXT, build_messages, parse_llm_answer
from app.services.embeddings import Embedder
from app.services.llm import LLMClient
from app.services.retrieval import retrieve

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    llm: LLMClient = Depends(get_llm),
):
    started = time.perf_counter()
    settings = get_settings()
    top_k = request.top_k or settings.final_top_k

    # Phase 2: only vector retrieval exists, so `retrieval`, `rerank` and `hyde` are accepted
    # but not used yet. `debug` reports what actually ran.
    result = retrieve(
        db, embedder, request.question, candidates=settings.retrieval_candidates, top_k=top_k
    )

    def respond(answer: str, found: bool, sources=None, citations_missing=False) -> AskResponse:
        return AskResponse(
            answer=answer,
            found=found,
            sources=sources or [],
            citations_missing=citations_missing,
            debug=AskDebug(
                retrieval="vector",
                rerank=False,
                hyde=False,
                candidates=result.candidates,
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    # Relevance gate: if even the best chunk is a weak match, do not call the LLM at all.
    # Cheaper, faster, and the model cannot make something up from irrelevant text.
    if not result.chunks or result.scores[0] < settings.vector_similarity_floor:
        return respond(NOT_FOUND_TEXT, found=False)

    try:
        llm_response = llm.complete(build_messages(request.question, result.chunks))
    except Exception as exc:
        logger.exception("LLM call failed")
        raise HTTPException(502, f"LLM request failed: {exc}")

    parsed = parse_llm_answer(llm_response.text, result.chunks)
    if not parsed.found:
        return respond(parsed.text, found=False)

    score_by_chunk = {chunk.id: score for chunk, score in zip(result.chunks, result.scores)}
    sources = [
        Source(
            file_name=chunk.file_name,
            page=chunk.page,
            score=round(score_by_chunk[chunk.id], 4),
            snippet=chunk.content[:200],
        )
        for chunk in parsed.used
    ]
    return respond(
        parsed.text, found=True, sources=sources, citations_missing=parsed.citations_missing
    )
