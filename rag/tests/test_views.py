from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from rag.models import ChatExchange, Document
from rag.tasks import ingest_document_task


class HomeViewTests(TestCase):
    def test_home_page_renders(self):
        response = self.client.get(reverse("rag:home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ask the indexed document set.")

    def test_document_detail_renders(self):
        document = Document.objects.create(title="Demo PDF", source_file="documents/demo.pdf")

        response = self.client.get(reverse("rag:document-detail", kwargs={"pk": document.pk}))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Demo PDF")

    def test_health_endpoint_returns_runtime_status(self):
        response = self.client.get(reverse("rag:health"))

        self.assertEqual(response.status_code, 200)
        self.assertIn("chat_model", response.json())

    def test_chat_post_creates_exchange_with_citations(self):
        match = MagicMock(
            metadata={
                "document_id": 1,
                "title": "Doc One",
                "page_number": 2,
                "chunk_index": 3,
                "text": "This is the supporting snippet for the answer.",
            },
            score=0.947,
        )

        with patch("rag.views.get_vector_store") as mock_vector_store, patch(
            "rag.views.get_chat_client"
        ) as mock_chat_client:
            mock_vector_store.return_value.query.return_value = [match]
            mock_vector_store.return_value.render_context.return_value = "context"
            mock_chat_client.return_value.answer.return_value = "Answer text"

            response = self.client.post(reverse("rag:chat"), {"question": "What is this?"})

        self.assertEqual(response.status_code, 302)
        self.assertEqual(ChatExchange.objects.count(), 1)
        exchange = ChatExchange.objects.get()
        self.assertEqual(exchange.answer, "Answer text")
        self.assertEqual(exchange.sources[0]["title"], "Doc One")
        self.assertEqual(exchange.sources[0]["page_number"], 2)
        self.assertEqual(exchange.sources[0]["chunk_index"], 3)
        self.assertEqual(exchange.sources[0]["score"], 0.947)
        self.assertIn("supporting snippet", exchange.sources[0]["snippet"])

    def test_document_detail_post_reindex_queues_task(self):
        document = Document.objects.create(title="Demo PDF", source_file="documents/demo.pdf")

        with patch("rag.views.ingest_document_task") as mock_task:
            response = self.client.post(
                reverse("rag:document-detail", kwargs={"pk": document.pk}),
                {"action": "reindex"},
            )

        self.assertEqual(response.status_code, 302)
        mock_task.delay.assert_called_once_with(document.id)

    def test_home_page_renders_saved_citations(self):
        ChatExchange.objects.create(
            session_key="session-1",
            question="What is in the document?",
            answer="Here is the grounded answer.",
            sources=[
                {
                    "title": "Doc One",
                    "page_number": 4,
                    "chunk_index": 0,
                    "has_chunk_index": True,
                    "score": 0.932,
                    "snippet": "Evidence snippet from the stored answer.",
                }
            ],
        )

        response = self.client.get(reverse("rag:home"))

        self.assertContains(response, "Doc One")
        self.assertContains(response, "Page 4")
        self.assertContains(response, "Chunk 0")
        self.assertContains(response, "Evidence snippet from the stored answer.")

    def test_document_detail_post_rejects_reindex_while_processing(self):
        document = Document.objects.create(
            title="Demo PDF",
            source_file="documents/demo.pdf",
            status=Document.Status.PROCESSING,
        )

        with patch("rag.views.ingest_document_task") as mock_task:
            response = self.client.post(
                reverse("rag:document-detail", kwargs={"pk": document.pk}),
                {"action": "reindex"},
                follow=True,
            )

        self.assertEqual(response.status_code, 200)
        mock_task.delay.assert_not_called()
        self.assertContains(response, "already processing")

    def test_document_detail_post_delete_removes_document_and_queues_cleanup(self):
        document = Document.objects.create(
            title="Demo PDF",
            source_file=SimpleUploadedFile("demo.pdf", b"%PDF-1.4\n"),
            pinecone_namespace="documents",
        )
        vector_id = f"doc-{document.id}-chunk-0"
        document.chunks.create(
            chunk_index=0,
            page_number=1,
            vector_id=vector_id,
            text_preview="chunk preview",
            metadata={},
        )
        document_id = document.id

        with patch("rag.views.delete_document_vectors_task") as mock_delete_task:
            response = self.client.post(
                reverse("rag:document-detail", kwargs={"pk": document.pk}),
                {"action": "delete"},
            )

        self.assertEqual(response.status_code, 302)
        self.assertFalse(Document.objects.filter(pk=document_id).exists())
        mock_delete_task.delay.assert_called_once_with(document_id, "documents", [vector_id])

    def test_document_detail_post_invalid_action_rejects_request(self):
        document = Document.objects.create(title="Demo PDF", source_file="documents/demo.pdf")

        response = self.client.post(
            reverse("rag:document-detail", kwargs={"pk": document.pk}),
            {"action": "nope"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid document action")

    def test_ingest_task_returns_missing_for_deleted_document(self):
        document = Document.objects.create(title="Demo PDF", source_file="documents/demo.pdf")
        document_id = document.id
        document.delete()

        result = ingest_document_task.run(document_id)

        self.assertEqual(result, {"document_id": document_id, "status": "missing"})

