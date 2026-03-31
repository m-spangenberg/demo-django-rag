from __future__ import annotations

from functools import lru_cache

from django.conf import settings
from langchain_ollama import OllamaEmbeddings


class SentenceTransformerClient:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embedding_client():
    if settings.EMBEDDING_BACKEND == "ollama":
        return OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return SentenceTransformerClient(settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_embedding_dimension() -> int:
    return len(get_embedding_client().embed_query("dimension probe"))
