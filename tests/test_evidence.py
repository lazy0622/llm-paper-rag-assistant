import unittest

from app.services.evidence import validate_citation_markers


class CitationValidationTests(unittest.TestCase):
    def test_accepts_existing_source_markers(self):
        self.assertEqual(validate_citation_markers("结论来自 [S1]，补充信息来自 [S2]。", 2), [])

    def test_reports_missing_markers(self):
        warnings = validate_citation_markers("这是一个没有结构化引用的回答。", 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("[S1]", warnings[0])

    def test_reports_unknown_source_markers(self):
        warnings = validate_citation_markers("结论来自 [S3]。", 2)
        self.assertEqual(len(warnings), 1)
        self.assertIn("[S3]", warnings[0])

    def test_no_sources_do_not_require_citations(self):
        self.assertEqual(validate_citation_markers("通用回答。", 0), [])


if __name__ == "__main__":
    unittest.main()
