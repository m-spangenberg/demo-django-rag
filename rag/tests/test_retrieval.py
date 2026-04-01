from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from rag.services.retrieval import PineconeVectorStore


class PineconeVectorStoreTests(SimpleTestCase):
    @override_settings(
        PINECONE_API_KEY="pclocal",
        PINECONE_URL="http://pinecone.local:5080",
        PINECONE_INDEX_HOST="http://pinecone.local:5081",
        PINECONE_INDEX="django-rag",
        PINECONE_METRIC="cosine",
    )
    @patch("rag.services.retrieval.get_embedding_dimension", return_value=768)
    @patch("rag.services.retrieval.Pinecone")
    def test_recreates_local_index_when_dimension_mismatch(self, mock_pinecone, mock_dimension):
        client = MagicMock()
        client.list_indexes.return_value = SimpleNamespace(names=lambda: ["django-rag"])
        client.describe_index.return_value = {"dimension": 384}
        mock_pinecone.return_value = client

        PineconeVectorStore()

        client.delete_index.assert_called_once_with(name="django-rag")
        client.create_index.assert_called_once()
        self.assertEqual(client.create_index.call_args.kwargs["dimension"], 768)

    @override_settings(
        PINECONE_API_KEY="real-key",
        PINECONE_URL="",
        PINECONE_INDEX_HOST="",
        PINECONE_INDEX="django-rag",
        PINECONE_METRIC="cosine",
        EMBEDDING_BACKEND="sentence-transformers",
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
    )
    @patch("rag.services.retrieval.get_embedding_dimension", return_value=384)
    @patch("rag.services.retrieval.Pinecone")
    def test_raises_for_remote_index_dimension_mismatch(self, mock_pinecone, mock_dimension):
        client = MagicMock()
        client.list_indexes.return_value = SimpleNamespace(names=lambda: ["django-rag"])
        client.describe_index.return_value = {"dimension": 768}
        mock_pinecone.return_value = client

        with self.assertRaises(RuntimeError):
            PineconeVectorStore()

    @override_settings(
        PINECONE_API_KEY="pclocal",
        PINECONE_URL="http://pinecone.local:5080",
        PINECONE_INDEX_HOST="http://pinecone.local:5081",
        PINECONE_INDEX="django-rag",
        PINECONE_METRIC="cosine",
        PINECONE_NAMESPACE="documents",
        RAG_TOP_K=2,
    )
    @patch("rag.services.retrieval.get_embedding_dimension", return_value=768)
    @patch("rag.services.retrieval.get_embedding_client")
    @patch("rag.services.retrieval.Pinecone")
    def test_query_merges_results_across_namespaces(self, mock_pinecone, mock_embedding_client, mock_dimension):
        client = MagicMock()
        client.list_indexes.return_value = SimpleNamespace(names=lambda: ["django-rag"])
        client.describe_index.return_value = {"dimension": 768}
        index = MagicMock()
        index.query.side_effect = [
            {
                "matches": [
                    {"id": "doc-1", "score": 0.41, "metadata": {"title": "First"}},
                ]
            },
            {
                "matches": [
                    {"id": "doc-2", "score": 0.93, "metadata": {"title": "Second"}},
                    {"id": "doc-3", "score": 0.72, "metadata": {"title": "Third"}},
                ]
            },
        ]
        client.Index.return_value = index
        mock_pinecone.return_value = client
        mock_embedding_client.return_value.embed_query.return_value = [0.1, 0.2, 0.3]

        store = PineconeVectorStore()

        matches = store.query(question="What changed?", namespaces=["ns-a", "ns-b"])

        self.assertEqual([match.id for match in matches], ["doc-2", "doc-3"])
        self.assertEqual(index.query.call_count, 2)

    @override_settings(
        PINECONE_API_KEY="pclocal",
        PINECONE_URL="http://pinecone.local:5080",
        PINECONE_INDEX_HOST="http://pinecone.local:5081",
        PINECONE_INDEX="django-rag",
        PINECONE_METRIC="cosine",
        PINECONE_NAMESPACE="documents",
        RAG_TOP_K=3,
    )
    @patch("rag.services.retrieval.get_embedding_dimension", return_value=768)
    @patch("rag.services.retrieval.get_embedding_client")
    @patch("rag.services.retrieval.Pinecone")
    def test_query_diversifies_results_across_documents(self, mock_pinecone, mock_embedding_client, mock_dimension):
        client = MagicMock()
        client.list_indexes.return_value = SimpleNamespace(names=lambda: ["django-rag"])
        client.describe_index.return_value = {"dimension": 768}
        index = MagicMock()
        index.query.return_value = {
            "matches": [
                {"id": "doc-1-a", "score": 0.99, "metadata": {"document_id": 1, "title": "Newest"}},
                {"id": "doc-1-b", "score": 0.98, "metadata": {"document_id": 1, "title": "Newest"}},
                {"id": "doc-1-c", "score": 0.97, "metadata": {"document_id": 1, "title": "Newest"}},
                {"id": "doc-2-a", "score": 0.86, "metadata": {"document_id": 2, "title": "Older"}},
                {"id": "doc-3-a", "score": 0.84, "metadata": {"document_id": 3, "title": "Oldest"}},
            ]
        }
        client.Index.return_value = index
        mock_pinecone.return_value = client
        mock_embedding_client.return_value.embed_query.return_value = [0.1, 0.2, 0.3]

        store = PineconeVectorStore()

        matches = store.query(question="Summarize the workspace")

        self.assertEqual([match.id for match in matches], ["doc-1-a", "doc-2-a", "doc-3-a"])