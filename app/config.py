from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Values come from environment variables (or a local .env file); names match .env.example.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://app:app@localhost:5432/docs"
    webhook_secret: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""  # may stay empty: tests use fakes and Ollama needs no key
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    chunk_size_chars: int = 2000
    chunk_overlap_chars: int = 300
    retrieval_candidates: int = 20
    final_top_k: int = 5
    rerank_score_threshold: float = 0.0
    max_upload_mb: int = 20
    vector_similarity_floor: float = 0.5  # below this cosine similarity we skip the LLM

    @field_validator("webhook_secret")
    @classmethod
    def secret_must_not_be_empty(cls, value: str) -> str:
        # Fail fast: an empty secret would make every signature trivially forgeable.
        if not value.strip():
            raise ValueError("WEBHOOK_SECRET must be set")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
