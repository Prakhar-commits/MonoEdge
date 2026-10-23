from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import models  # noqa: F401  (registers tables on Base)
from app.config import get_settings
from app.db import Base, engine, get_db
from app.routers import ask, documents, webhooks


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fails fast if WEBHOOK_SECRET is empty
    # No Alembic for this project: tables are created on startup (noted in README).
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="AI Docs Assistant", lifespan=lifespan)
app.include_router(documents.router)
app.include_router(ask.router)
app.include_router(webhooks.router)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok"}
