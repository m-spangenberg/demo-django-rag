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