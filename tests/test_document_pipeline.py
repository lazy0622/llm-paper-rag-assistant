import tempfile
import unittest
from pathlib import Path

from app.services.document_loader import load_document
from app.services.splitter import split_pages


class DocumentPipelineTests(unittest.TestCase):
    def test_text_document_preserves_identity_and_context_metadata(self):
        content = "Introduction\nRAG combines retrieval with generation.\n\nThe index stores evidence."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.md"
            path.write_text(content, encoding="utf-8")

            pages = load_document(path, file_name="paper.md")
            chunks = split_pages(pages)

        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["document_title"], "paper")
        self.assertEqual(pages[0]["section"], "Introduction")
        self.assertTrue(pages[0]["content_hash"])
        self.assertGreater(len(chunks), 0)
        self.assertTrue(chunks[0]["chunk_id"])
        self.assertEqual(chunks[0]["document_title"], "paper")
        self.assertTrue(chunks[0]["parent_chunk_id"])
        self.assertIn("Document title: paper", chunks[0]["embedding_text"])


if __name__ == "__main__":
    unittest.main()
