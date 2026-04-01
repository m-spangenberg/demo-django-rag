from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypedDict, cast

from django.conf import settings
from pinecone import Pinecone, ServerlessSpec

from .embeddings import get_embedding_client, get_embedding_dimension


@dataclass(slots=True)
class RetrievalMatch:
    id: str
    score: float
    metadata: dict


class VectorRecord(TypedDict):
    id: str
    values: list[float]
    metadata: dict[str, Any]


class PineconeVectorStore:
    def __init__(self):
        if not settings.PINECONE_API_KEY and not settings.PINECONE_URL:
            raise RuntimeError("PINECONE_API_KEY is not configured.")
        client_kwargs = {"api_key": settings.PINECONE_API_KEY or "pclocal"}
        if settings.PINECONE_URL:
            client_kwargs["host"] = settings.PINECONE_URL
        self.client = Pinecone(**client_kwargs)
        self.index_name = settings.PINECONE_INDEX
        self.dimension = get_embedding_dimension()
        listed_indexes = self.client.list_indexes()
        if hasattr(listed_indexes, "names"):
            existing_indexes = set(listed_indexes.names())
        else:
            existing_indexes = {
                item["name"] if isinstance(item, dict) else item.name for item in listed_indexes
            }
        if self.index_name not in existing_indexes:
            self._create_index()
        else:
            self._ensure_index_dimension()
        index_host = settings.PINECONE_INDEX_HOST or ""
        if index_host:
            self.index = self.client.Index(host=index_host)
        else:
            self.index = self.client.Index(self.index_name)

    def _create_index(self) -> None:
        self.client.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    def _ensure_index_dimension(self) -> None:
        current_dimension = self._get_index_dimension()
        if current_dimension is None or current_dimension == self.dimension:
            return
        if self._uses_local_pinecone():
            self.client.delete_index(name=self.index_name)
            self._create_index()
            return
        raise RuntimeError(
            f"Pinecone index '{self.index_name}' has dimension {current_dimension}, "
            f"but embedding model '{settings.OLLAMA_EMBED_MODEL if settings.EMBEDDING_BACKEND == 'ollama' else settings.EMBEDDING_MODEL}' "
            f"produces {self.dimension}. Update PINECONE_DIMENSION and recreate the index."
        )

    def _get_index_dimension(self) -> int | None:
        description = self.client.describe_index(name=self.index_name)
        if isinstance(description, dict):
            dimension = description.get("dimension")
            return int(dimension) if dimension is not None else None
        if hasattr(description, "dimension"):
            dimension = getattr(description, "dimension")
            return int(dimension) if dimension is not None else None
        if hasattr(description, "to_dict"):
            payload = description.to_dict()
            dimension = payload.get("dimension")
            return int(dimension) if dimension is not None else None
        return None

    def _uses_local_pinecone(self) -> bool:
        targets = " ".join(
            value.lower()
            for value in (settings.PINECONE_URL, settings.PINECONE_INDEX_HOST)
            if value
        )
        return any(marker in targets for marker in ("pinecone.local", "localhost", "127.0.0.1"))

    def upsert(self, *, records: list[VectorRecord], namespace: str) -> None:
        self.index.upsert(vectors=cast(Any, records), namespace=namespace)

    def delete_document_vectors(
        self,
        *,
        namespace: str,
        vector_ids: list[str] | None = None,
    ) -> None:
        if not vector_ids:
            return
        self.index.delete(ids=vector_ids, namespace=namespace)

    def _select_diverse_matches(
        self,
        *,
        matches: list[RetrievalMatch],
        limit: int,
    ) -> list[RetrievalMatch]:
        if len(matches) <= limit:
            return matches

        selected: list[RetrievalMatch] = []
        per_document_counts: dict[object, int] = {}
        max_per_document = 1

        while len(selected) < limit:
            added_in_round = False
            for match in matches:
                if match in selected:
                    continue
                document_id = match.metadata.get("document_id") or match.id
                if per_document_counts.get(document_id, 0) >= max_per_document:
                    continue
                selected.append(match)
                per_document_counts[document_id] = per_document_counts.get(document_id, 0) + 1
                added_in_round = True
                if len(selected) == limit:
                    break
            if not added_in_round:
                max_per_document += 1

        return selected

    def query(
        self,
        *,
        question: str,
        top_k: int | None = None,
        namespaces: list[str] | None = None,
    ) -> list[RetrievalMatch]:
        embedding = get_embedding_client().embed_query(question)
        resolved_top_k = top_k or settings.RAG_TOP_K
        candidate_top_k = max(resolved_top_k * 4, resolved_top_k)
        query_namespaces = namespaces or [settings.PINECONE_NAMESPACE]

        all_matches: list[RetrievalMatch] = []
        for namespace in dict.fromkeys(query_namespaces):
            response = self.index.query(
                vector=embedding,
                top_k=candidate_top_k,
                include_metadata=True,
                namespace=namespace,
            )
            if isinstance(response, dict):
                matches = cast(dict[str, Any], response).get("matches", [])
            else:
                matches = getattr(response, "matches", [])
            all_matches.extend(
                RetrievalMatch(
                    id=cast(str, match["id"]),
                    score=float(match["score"]),
                    metadata=cast(dict[str, Any], match.get("metadata", {})),
                )
                for match in matches
            )

        all_matches.sort(key=lambda match: match.score, reverse=True)
        return self._select_diverse_matches(matches=all_matches, limit=resolved_top_k)

    def render_context(self, matches: list[RetrievalMatch]) -> str:
        parts: list[str] = []
        total_chars = 0
        for match in matches:
            title = match.metadata.get("title", "Untitled")
            page = match.metadata.get("page_number", "?")
            text = str(match.metadata.get("text", "")).strip()
            if not text:
                continue
            snippet = f"[{title} p.{page}] {text}"
            if total_chars + len(snippet) > settings.RAG_MAX_CONTEXT_CHARS:
                break
            total_chars += len(snippet)
            parts.append(snippet)
        return "\n\n".join(parts)


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    return PineconeVectorStore()
