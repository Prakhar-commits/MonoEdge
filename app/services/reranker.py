from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedReranker:
    """Cross-encoder: reads the question and a chunk together and scores how well they match."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._get_model().rerank(query, texts)]
