"""Compare retrieval configurations on the sample documents and write eval/results.md.

Usage (the database from `docker compose up db -d` must be running):
    python eval/make_sample_docs.py     # only needed if eval/docs is missing
    python eval/run_eval.py

It uses real models (embedding model, cross-encoder, and the LLM for the HyDE config) and a
SEPARATE database called docs_eval, so your real uploaded documents are never touched.
Retrieval is measured directly; the LLM is only used for HyDE, never to write answers.
"""
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The app reads its settings when it is imported, so point it at the eval database first.
EVAL_DATABASE_URL = os.environ.get(
    "EVAL_DATABASE_URL", "postgresql+psycopg://app:app@localhost:5433/docs_eval"
)
os.environ["DATABASE_URL"] = EVAL_DATABASE_URL
os.environ.setdefault("WEBHOOK_SECRET", "eval-not-used")

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from app import models  # noqa: E402, F401
from app.config import get_settings  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.deps import get_embedder, get_llm, get_reranker  # noqa: E402
from app.services.ingestion import extract_pages, ingest_document  # noqa: E402
from app.services.retrieval import retrieve  # noqa: E402

DOCS_DIR = ROOT / "eval" / "docs"
QUESTIONS_FILE = ROOT / "eval" / "questions.json"
RESULTS_FILE = ROOT / "eval" / "results.md"
TOP_K = 5

CONFIGS = [
    ("vector", {"retrieval": "vector", "rerank": False, "hyde": False}),
    ("keyword", {"retrieval": "keyword", "rerank": False, "hyde": False}),
    ("hybrid", {"retrieval": "hybrid", "rerank": False, "hyde": False}),
    ("hybrid+rerank", {"retrieval": "hybrid", "rerank": True, "hyde": False}),
    ("hybrid+rerank+hyde", {"retrieval": "hybrid", "rerank": True, "hyde": True}),
]


def create_database_if_missing() -> None:
    url = make_url(EVAL_DATABASE_URL)
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        exists = conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": url.database}
        )
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    admin.dispose()


def prepare_database() -> None:
    create_database_if_missing()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents CASCADE"))


def ingest_sample_docs(embedder) -> None:
    settings = get_settings()
    files = sorted(p for p in DOCS_DIR.iterdir() if p.suffix in {".pdf", ".txt", ".md"})
    if not files:
        sys.exit("No files in eval/docs. Run: python eval/make_sample_docs.py")
    with SessionLocal() as db:
        for path in files:
            data = path.read_bytes()
            result = ingest_document(
                db,
                embedder,
                file_name=path.name,
                content_type="application/octet-stream",
                sha256=hashlib.sha256(data).hexdigest(),
                pages=extract_pages(data, path.suffix),
                chunk_size=settings.chunk_size_chars,
                chunk_overlap=settings.chunk_overlap_chars,
            )
            print(f"ingested {path.name}: {result.chunk_count} chunks")


def first_match_rank(chunks, question) -> int | None:
    """1-based rank of the first chunk from the expected (file, page), or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if (chunk.file_name, chunk.page) == (question["expected_file"], question["expected_page"]):
            return rank
    return None


def run_config(services, questions, options) -> list[dict]:
    settings = get_settings()
    embedder, reranker, llm = services
    rows = []
    with SessionLocal() as db:
        for question in questions:
            started = time.perf_counter()
            result = retrieve(
                db,
                embedder,
                reranker,
                llm,
                question["question"],
                top_k=TOP_K,
                candidates=settings.retrieval_candidates,
                similarity_floor=settings.vector_similarity_floor,
                rerank_threshold=settings.rerank_score_threshold,
                **options,
            )
            rows.append(
                {
                    "question": question,
                    "rank": first_match_rank(result.chunks, question),
                    "gate_passed": result.gate_passed,
                    "top_score": result.top_score,
                    "ms": (time.perf_counter() - started) * 1000,
                }
            )
    return rows


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["question"]["answerable"]]
    unanswerable = [r for r in rows if not r["question"]["answerable"]]

    def hit_at(k, subset):
        return sum(1 for r in subset if r["rank"] is not None and r["rank"] <= k)

    def hit1_by_kind(kind):
        subset = [r for r in answerable if r["question"]["kind"] == kind]
        return f"{hit_at(1, subset)}/{len(subset)}"

    return {
        "hit1": f"{hit_at(1, answerable)}/{len(answerable)}",
        "hit5": f"{hit_at(5, answerable)}/{len(answerable)}",
        "mrr": statistics.mean(1 / r["rank"] if r["rank"] else 0 for r in answerable),
        "exact_hit1": hit1_by_kind("exact"),
        "paraphrase_hit1": hit1_by_kind("paraphrase"),
        "gate_ok": f"{sum(1 for r in unanswerable if not r['gate_passed'])}/{len(unanswerable)}",
        "false_reject": f"{sum(1 for r in answerable if not r['gate_passed'])}/{len(answerable)}",
        "ms": statistics.mean(r["ms"] for r in rows),
    }


def markdown_report(summaries: dict, all_rows: dict, skipped: list[str]) -> str:
    lines = [
        "# Retrieval evaluation",
        "",
        f"10 questions (8 answerable, 2 unanswerable) over the sample documents, top_k={TOP_K}, "
        f"chunk size {get_settings().chunk_size_chars}, "
        f"{get_settings().retrieval_candidates} candidates per search.",
        "",
        "| Config | hit@1 | hit@5 | MRR | exact hit@1 | paraphrase hit@1 | "
        "unanswerable rejected | answerable wrongly rejected | mean ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['hit1']} | {s['hit5']} | {s['mrr']:.2f} | {s['exact_hit1']} | "
            f"{s['paraphrase_hit1']} | {s['gate_ok']} | {s['false_reject']} | {s['ms']:.0f} |"
        )
    for name in skipped:
        lines.append(f"| {name} | skipped (no LLM key) | | | | | | | |")

    lines += [
        "",
        "Gate for configs without rerank: best vector cosine similarity must reach "
        f"{get_settings().vector_similarity_floor}; keyword-only has no similarity, so any hit "
        "passes. Gate for rerank configs: best cross-encoder score must reach "
        f"{get_settings().rerank_score_threshold}.",
        "",
        "## Score calibration",
        "",
        "| Question | Kind | Top vector cosine | Top rerank score |",
        "|---|---|---|---|",
    ]
    vector_rows = all_rows["vector"]
    rerank_rows = all_rows["hybrid+rerank"]
    for v, r in zip(vector_rows, rerank_rows):
        lines.append(
            f"| {v['question']['question']} | {v['question']['kind']} | "
            f"{v['top_score']:.3f} | {r['top_score']:.2f} |"
        )
    lines += ["", calibration_advice(vector_rows, rerank_rows), ""]

    names = list(all_rows)
    lines += [
        "## Rank of the correct page per question (blank = not in top 5)",
        "",
        "| Question | Kind | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for index, row in enumerate(all_rows[names[0]]):
        if row["question"]["answerable"]:
            ranks = " | ".join(str(all_rows[n][index]["rank"] or "") for n in names)
            lines.append(f"| {row['question']['question']} | {row['question']['kind']} | {ranks} |")
    lines.append("")
    return "\n".join(lines)


def calibration_advice(vector_rows: list[dict], rerank_rows: list[dict]) -> str:
    def split(rows):
        yes = [r["top_score"] for r in rows if r["question"]["answerable"]]
        no = [r["top_score"] for r in rows if not r["question"]["answerable"]]
        return min(yes), max(no)

    v_low, v_high = split(vector_rows)
    r_low, r_high = split(rerank_rows)
    text_lines = [
        f"- Lowest score of an answerable question vs highest of an unanswerable one: "
        f"cosine {v_low:.3f} vs {v_high:.3f}; rerank {r_low:.2f} vs {r_high:.2f}.",
    ]
    for label, low, high in (("VECTOR_SIMILARITY_FLOOR", v_low, v_high),
                             ("RERANK_SCORE_THRESHOLD", r_low, r_high)):
        if low > high:
            text_lines.append(
                f"- The two groups separate cleanly. Any {label} between {high:.2f} and "
                f"{low:.2f} works; the midpoint is {(low + high) / 2:.2f}."
            )
        else:
            text_lines.append(f"- {label}: the groups overlap, so no single value is perfect.")
    return "\n".join(text_lines)


def main() -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    prepare_database()
    embedder, reranker, llm = get_embedder(), get_reranker(), get_llm()

    # Load the models once up front so the first config is not charged for the load time.
    embedder.embed_query("warm up")
    reranker.score("warm up", ["warm up"])
    ingest_sample_docs(embedder)

    has_llm_key = bool(get_settings().llm_api_key)
    summaries, all_rows, skipped = {}, {}, []
    for name, options in CONFIGS:
        if options["hyde"] and not has_llm_key:
            skipped.append(name)
            continue
        print(f"running {name} ...")
        rows = run_config((embedder, reranker, llm), questions, options)
        all_rows[name] = rows
        summaries[name] = summarise(rows)

    report = markdown_report(summaries, all_rows, skipped)
    RESULTS_FILE.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"\nWrote {RESULTS_FILE}")


if __name__ == "__main__":
    main()
