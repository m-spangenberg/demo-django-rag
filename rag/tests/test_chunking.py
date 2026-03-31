from django.test import SimpleTestCase

from rag.services.chunking import PageText, build_chunks


class ChunkingTests(SimpleTestCase):
    def test_build_chunks_splits_long_text_with_overlap(self):
        pages = [PageText(page_number=1, text="abcdefghij" * 3)]

        chunks = build_chunks(pages, chunk_size=10, chunk_overlap=2)

        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].text, "abcdefghij")
