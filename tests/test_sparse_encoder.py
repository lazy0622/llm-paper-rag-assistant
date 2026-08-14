import unittest

from app.services.sparse_encoder import encode_sparse, tokenize_sparse


class SparseEncoderTests(unittest.TestCase):
    def test_tokenization_keeps_identifiers_and_chinese_terms(self):
        terms = tokenize_sparse("RAG-Sequence 与向量数据库 Qdrant")
        self.assertIn("rag-sequence", terms)
        self.assertIn("向量数据库", terms)
        self.assertIn("qdrant", terms)

    def test_encoding_is_deterministic_and_sorted(self):
        first = encode_sparse("RAG RAG hybrid search")
        second = encode_sparse("RAG RAG hybrid search")
        self.assertEqual(first, second)
        self.assertEqual(first.indices, sorted(first.indices))
        self.assertEqual(len(first.indices), len(first.values))
        self.assertGreater(len(first.indices), 0)


if __name__ == "__main__":
    unittest.main()
