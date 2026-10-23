from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embedding model. The model is loaded once, on first use, not per request."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: slow import, big download

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._get_model().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._get_model().query_embed(text))).tolist()
