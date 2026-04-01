from __future__ import annotations

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views import View

from .forms import ChatQueryForm, DocumentActionForm, HomeActionForm, UploadDocumentForm
from .models import ChatExchange, Document
from .services.llm import get_chat_client, get_runtime_status
from .services.retrieval import RetrievalMatch, get_vector_store
from .tasks import delete_document_vectors_task, get_document_vector_ids, ingest_document_task


def serialize_sources(matches: list[RetrievalMatch]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for match in matches:
        metadata = match.metadata
        text = str(metadata.get("text", "")).strip()
        serialized.append(
            {
                "document_id": metadata.get("document_id"),
                "title": metadata.get("title", "Untitled"),
                "page_number": metadata.get("page_number", "?"),
                "chunk_index": metadata.get("chunk_index"),
                "has_chunk_index": metadata.get("chunk_index") is not None,
                "score": round(match.score, 3),
                "snippet": text[:220],
            }
        )
    return serialized


class HomeView(View):
    template_name = "rag/home.html"

    def get(self, request):
        context = {
            "documents": Document.objects.all()[:10],
            "upload_form": UploadDocumentForm(),
            "chat_form": ChatQueryForm(),
            "exchanges": ChatExchange.objects.all()[:10],
            "runtime_status": get_runtime_status(),
        }
        return render(request, self.template_name, context)

    def post(self, request):
        if request.POST.get("action"):
            action_form = HomeActionForm(request.POST)
            if not action_form.is_valid():
                messages.error(request, "Invalid home action.")
                return redirect("rag:home")

            ChatExchange.objects.all().delete()
            messages.success(request, "Recent answer feed cleared.")
            return redirect("rag:home")

        form = UploadDocumentForm(request.POST, request.FILES)
        if not form.is_valid():
            context = {
                "documents": Document.objects.all()[:10],
                "upload_form": form,
                "chat_form": ChatQueryForm(),
                "exchanges": ChatExchange.objects.all()[:10],
                "runtime_status": get_runtime_status(),
            }
            return render(request, self.template_name, context, status=400)

        document = Document.objects.create(
            title=form.cleaned_data["title"],
            source_file=form.cleaned_data["pdf"],
        )
        ingest_document_task.delay(document.id)
        messages.success(request, "Document uploaded and queued for processing.")
        return redirect(reverse("rag:document-detail", kwargs={"pk": document.pk}))


class DocumentDetailView(View):
    template_name = "rag/document_detail.html"

    def get(self, request, pk: int):
        document = get_object_or_404(Document, pk=pk)
        context = {
            "document": document,
            "chunks": document.chunks.all()[:25],
        }
        return render(request, self.template_name, context)

    def post(self, request, pk: int):
        document = get_object_or_404(Document, pk=pk)
        form = DocumentActionForm(request.POST)
        if not form.is_valid():
            messages.error(request, "Invalid document action.")
            return redirect("rag:document-detail", pk=pk)

        action = form.cleaned_data["action"]
        if action == "reindex":
            if document.status == Document.Status.PROCESSING:
                messages.error(request, "This document is already processing.")
                return redirect("rag:document-detail", pk=pk)
            ingest_document_task.delay(document.id)
            messages.success(request, "Document queued for re-indexing.")
            return redirect("rag:document-detail", pk=pk)

        namespace = document.pinecone_namespace or ""
        document_id = document.id
        vector_ids = get_document_vector_ids(document)
        storage = document.source_file.storage
        source_name = document.source_file.name

        if namespace:
            delete_document_vectors_task.delay(document_id, namespace, vector_ids)

        if source_name:
            storage.delete(source_name)
        document.delete()
        messages.success(request, "Document deleted.")
        return redirect("rag:home")


class ChatView(View):
    def post(self, request):
        form = ChatQueryForm(request.POST)
        if not form.is_valid():
            messages.error(request, "Enter a question before querying the knowledge base.")
            return redirect("rag:home")

        question = form.cleaned_data["question"]
        try:
            vector_store = get_vector_store()
            matches = vector_store.query(question=question)
            context_text = vector_store.render_context(matches)
            chat_client = get_chat_client()
            answer = chat_client.answer(question=question, context=context_text)
        except Exception as exc:
            messages.error(request, f"Query failed: {exc}")
            return redirect("rag:home")

        if not request.session.session_key:
            request.session.create()

        ChatExchange.objects.create(
            session_key=request.session.session_key,
            question=question,
            answer=answer,
            sources=serialize_sources(matches),
        )
        return redirect("rag:home")


def document_health(request):
    return JsonResponse(get_runtime_status())
