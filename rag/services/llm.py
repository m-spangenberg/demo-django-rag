from __future__ import annotations

from functools import lru_cache

import requests
from django.conf import settings
from langchain_ollama import ChatOllama


class LocalChatClient:
    def __init__(self, model: str, base_url: str):
        self.client = ChatOllama(model=model, base_url=base_url, temperature=0)

    def answer(self, *, question: str, context: str) -> str:
        prompt = (
            "You are a retrieval-augmented assistant. Answer using only the provided context. "
            "If the answer is not in the context, say that directly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = self.client.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)


@lru_cache(maxsize=1)
def get_chat_client() -> LocalChatClient:
    return LocalChatClient(
        model=settings.OLLAMA_CHAT_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def get_runtime_status() -> dict[str, object]:
    payload = {
        "chat_model": settings.OLLAMA_CHAT_MODEL,
        "embedding_backend": settings.EMBEDDING_BACKEND,
        "embedding_model": settings.OLLAMA_EMBED_MODEL
        if settings.EMBEDDING_BACKEND == "ollama"
        else settings.EMBEDDING_MODEL,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "ollama_reachable": False,
        "inference_mode": "unknown",
    }
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
        payload["ollama_reachable"] = True
        payload["inference_mode"] = "gpu_or_cpu_managed_by_ollama"
    except requests.RequestException:
        payload["inference_mode"] = "unavailable"
    return payload
