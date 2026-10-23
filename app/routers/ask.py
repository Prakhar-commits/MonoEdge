import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.deps import get_embedder, get_llm, get_reranker
from app.schemas import AskDebug, AskRequest, AskResponse, Source, TicketOut
from app.services.agent import try_create_ticket, wants_ticket
from app.services.answer import NOT_FOUND_TEXT, build_messages, parse_llm_answer
from app.services.embeddings import Embedder
from app.services.llm import LLMClient
from app.services.reranker import Reranker
from app.services.retrieval import retrieve

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    db: Session = Depends(get_db),
    embedder: Embedder = Depends(get_embedder),
    reranker: Reranker = Depends(get_reranker),
    llm: LLMClient = Depends(get_llm),
):
    started = time.perf_counter()
    settings = get_settings()

    # "Raise a ticket ..." requests go to the tool-calling agent first; if the model does
    # not call the tool, we carry on with normal document question answering.
    if wants_ticket(request.question):
        agent_result = try_create_ticket(db, llm, request.question)
        if agent_result is not None:
            action = None
            if agent_result.ticket is not None:
                ticket = TicketOut.model_validate(agent_result.ticket).model_dump(mode="json")
                action = {"type": "ticket_created", "ticket": ticket}
            return AskResponse(
                answer=agent_result.answer,
                found=agent_result.found,
                sources=[],
                action=action,
                debug=AskDebug(
                    retrieval="agent",
                    rerank=False,
                    hyde=False,
                    candidates=0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                ),
            )

    result = retrieve(
        db,
        embedder,
        reranker,
        llm,
        request.question,
        retrieval=request.retrieval,
        rerank=request.rerank,
        hyde=request.hyde,
        top_k=request.top_k or settings.final_top_k,
        candidates=settings.retrieval_candidates,
        similarity_floor=settings.vector_similarity_floor,
        rerank_threshold=settings.rerank_score_threshold,
    )

    def respond(answer: str, found: bool, sources=None, citations_missing=False) -> AskResponse:
        return AskResponse(
            answer=answer,
            found=found,
            sources=sources or [],
            citations_missing=citations_missing,
            debug=AskDebug(
                retrieval=request.retrieval,
                rerank=request.rerank,
                hyde=result.hyde_used,
                candidates=result.candidates,
                top_score=None if result.top_score is None else round(result.top_score, 4),
                latency_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    # Relevance gate: if the best match is weak, do not call the LLM at all. Cheaper, faster,
    # and the model cannot make something up from irrelevant text.
    if not result.gate_passed:
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
