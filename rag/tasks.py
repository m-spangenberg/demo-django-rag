from __future__ import annotations

from celery import shared_task
from django.conf import settings

from .models import Document, DocumentChunk
from .services.chunking import build_chunks
from .services.embeddings import get_embedding_client
from .services.pdf import extract_pdf_pages
from .services.retrieval import get_vector_store


def get_document_vector_ids(document: Document) -> list[str]:
    vector_ids = list(document.chunks.values_list("vector_id", flat=True))
    if vector_ids or not document.chunk_count:
        return vector_ids
    document_id = document.pk
    if document_id is None:
        return []
    return [f"doc-{document_id}-chunk-{index}" for index in range(document.chunk_count)]


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def ingest_document_task(self, document_id: int) -> dict[str, int | str]:
    try:
        document = Document.objects.get(pk=document_id)
    except Document.DoesNotExist:
        return {"document_id": document_id, "status": "missing"}
    document_pk = document.pk
    if document_pk is None:
        raise ValueError("Document must exist before ingestion starts.")
    document.mark_processing()

    vector_store = get_vector_store()
    embedding_client = get_embedding_client()

    try:
        pages = extract_pdf_pages(document.source_file.path)
        chunks = build_chunks(
            pages,
            chunk_size=settings.RAG_CHUNK_SIZE,
            chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("No extractable text was found in the PDF.")

        namespace = document.pinecone_namespace or settings.PINECONE_NAMESPACE
        existing_vector_ids = get_document_vector_ids(document)
        vector_store.delete_document_vectors(namespace=namespace, vector_ids=existing_vector_ids)
        DocumentChunk.objects.filter(document=document).delete()

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_client.embed_documents(texts)

        if not Document.objects.filter(pk=document_pk).exists():
            return {"document_id": document_id, "status": "deleted"}

        records = []
        chunk_models = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            vector_id = f"doc-{document_pk}-chunk-{index}"
            metadata = {
                "document_id": document_pk,
                "title": document.title,
                "page_number": chunk.page_number,
                "chunk_index": index,
                "text": chunk.text,
            }
            records.append({"id": vector_id, "values": embedding, "metadata": metadata})
            chunk_models.append(
                DocumentChunk(
                    document=document,
                    chunk_index=index,
                    page_number=chunk.page_number,
                    vector_id=vector_id,
                    text_preview=chunk.text[:500],
                    metadata=metadata,
                )
            )

        if not Document.objects.filter(pk=document_pk).exists():
            return {"document_id": document_id, "status": "deleted"}

        vector_store.upsert(records=records, namespace=namespace)
        DocumentChunk.objects.bulk_create(chunk_models)
        refreshed_document = Document.objects.filter(pk=document_pk).first()
        if refreshed_document is None:
            return {"document_id": document_id, "status": "deleted"}
        refreshed_document.mark_ready(chunk_count=len(chunks), page_count=len(pages), namespace=namespace)
        return {"document_id": document_pk, "chunks": len(chunks), "pages": len(pages)}
    except Exception as exc:
        refreshed_document = Document.objects.filter(pk=document_id).first()
        if refreshed_document is not None:
            refreshed_document.mark_failed(str(exc))
        raise


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def delete_document_vectors_task(
    self,
    document_id: int,
    namespace: str,
    vector_ids: list[str] | None = None,
) -> dict[str, int | str]:
    vector_store = get_vector_store()
    vector_store.delete_document_vectors(namespace=namespace, vector_ids=vector_ids)
    return {
        "document_id": document_id,
        "namespace": namespace,
        "deleted_vectors": len(vector_ids or []),
    }
