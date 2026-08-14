import unittest

from app.services.evaluation import answer_overlap_metrics, parse_expected_pages, retrieval_metrics


class EvaluationMetricsTests(unittest.TestCase):
    def test_answer_overlap_supports_chinese_and_english(self):
        metrics = answer_overlap_metrics(
            "参数化记忆难以及时更新，且不能提供可追溯依据",
            "参数化记忆中的知识难以及时更新，也难以提供可追溯的来源依据。",
        )

        self.assertGreater(metrics["answer_token_recall"], 0.5)
        self.assertGreater(metrics["answer_token_f1"], 0.5)

    def test_retrieval_prefers_exact_chunk_ground_truth(self):
        metrics = retrieval_metrics(
            [
                {"chunk_id": "wrong", "file_name": "paper.pdf", "page": 2},
                {"chunk_id": "gold", "file_name": "paper.pdf", "page": 3},
            ],
            expected_source_files=["paper.pdf"],
            gold_chunk_ids=["gold"],
            top_k=2,
        )

        self.assertTrue(metrics["source_hit"])
        self.assertEqual(metrics["retrieval_precision_at_k"], 0.5)
        self.assertEqual(metrics["retrieval_recall_at_k"], 1.0)
        self.assertEqual(metrics["retrieval_mrr"], 0.5)
        self.assertEqual(metrics["ground_truth_level"], "chunk")

    def test_retrieval_can_use_file_and_page_ground_truth(self):
        metrics = retrieval_metrics(
            [{"chunk_id": "c1", "file_name": "paper.pdf", "page": 4}],
            expected_source_files=["paper.pdf"],
            gold_pages=[4],
        )

        self.assertEqual(metrics["ground_truth_level"], "page")
        self.assertEqual(metrics["retrieval_recall_at_k"], 1.0)

    def test_parse_expected_pages_ignores_invalid_values(self):
        self.assertEqual(parse_expected_pages("3;bad|5;0|-1"), [3, 5])


if __name__ == "__main__":
    unittest.main()
