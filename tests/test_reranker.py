import unittest
from unittest.mock import patch

from app.config import settings
from app.services import reranker


class _FakeCrossEncoder:
    def predict(self, pairs, batch_size, show_progress_bar):
        self.last_call = (pairs, batch_size, show_progress_bar)
        return [2.0, -2.0]


class RerankerTests(unittest.TestCase):
    def test_cross_encoder_provider_reranks_candidates(self):
        semantic = [
            {
                "chunk_id": "a",
                "content": "first candidate",
                "score": 0.9,
                "semantic_rank": 1,
                "keyword_score": 0.0,
                "keyword_rank": None,
                "retrieval_source": "semantic",
            },
            {
                "chunk_id": "b",
                "content": "second candidate",
                "score": 0.8,
                "semantic_rank": 2,
                "keyword_score": 0.0,
                "keyword_rank": None,
                "retrieval_source": "semantic",
            },
        ]
        fake_model = _FakeCrossEncoder()
        with patch.object(settings, "reranker_provider", "cross_encoder"), patch.object(
            settings, "reranker_model", "fake-model"
        ), patch.object(settings, "reranker_batch_size", 4), patch.object(
            reranker, "_get_cross_encoder", return_value=fake_model
        ):
            result = reranker.rerank_candidates(semantic, [], query="target query")

        self.assertEqual([chunk["chunk_id"] for chunk in result], ["a", "b"])
        self.assertEqual(result[0]["reranker_provider"], "cross_encoder")
        self.assertGreater(result[0]["rerank_score"], result[1]["rerank_score"])
        self.assertEqual(fake_model.last_call[1:], (4, False))

    def test_unknown_provider_fails_explicitly(self):
        with patch.object(settings, "reranker_provider", "unknown"):
            with self.assertRaisesRegex(ValueError, "Expected 'rule' or 'cross_encoder'"):
                reranker.rerank_candidates([], [])

    def test_rule_provider_remains_default_and_does_not_load_model(self):
        with patch.object(settings, "reranker_provider", "rule"), patch.object(
            reranker, "_get_cross_encoder", side_effect=AssertionError("must not load model")
        ):
            result = reranker.rerank_candidates(
                [{"chunk_id": "a", "content": "candidate", "score": 0.8, "semantic_rank": 1}],
                [],
            )

        self.assertEqual(result[0]["reranker_provider"], "rule")


if __name__ == "__main__":
    unittest.main()
