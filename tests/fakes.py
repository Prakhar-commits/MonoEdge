import hashlib
import math
import re

from app.models import EMBEDDING_DIMS
from app.services.llm import LLMResponse


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbedder:
    """Deterministic bag-of-words embedding: shared words give higher cosine similarity."""

    def vector(self, text: str) -> list[float]:
        vec = [0.0] * EMBEDDING_DIMS
        for token in tokens(text) or ["empty"]:
            slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % EMBEDDING_DIMS
            vec[slot] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self.vector(t) for t in texts]

    def embed_query(self, text):
        return self.vector(text)


class FakeReranker:
    """Score = fraction of the query's words that appear in the text. Records its calls."""

    def __init__(self):
        self.queries = []

    def score(self, query, texts):
        self.queries.append(query)
        query_words = set(tokens(query))
        return [len(query_words & set(tokens(t))) / max(len(query_words), 1) for t in texts]


class FakeLLM:
    """Returns a scripted reply and records every call so tests can assert on it."""

    def __init__(self, reply: str = "NOT_FOUND", tool_calls=None):
        self.reply = reply
        self.tool_calls = tool_calls or []  # scripted tool calls for agent tests
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return LLMResponse(text=self.reply, tool_calls=list(self.tool_calls))
