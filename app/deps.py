"""Dependencies that return the embedder / reranker / LLM.

Routes take these with Depends(...), so tests can swap in fakes with
app.dependency_overrides and never download a model or call an API.
"""
from functools import lru_cache

from app.config import get_settings
from app.services.embeddings import Embedder, FastEmbedEmbedder
from app.services.llm import LLMClient, OpenAICompatibleLLM
from app.services.reranker import FastEmbedReranker, Reranker


@lru_cache
def get_embedder() -> Embedder:
    return FastEmbedEmbedder(get_settings().embedding_model)


@lru_cache
def get_reranker() -> Reranker:
    return FastEmbedReranker(get_settings().reranker_model)


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
