import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import ingestion_jobs
from app.services.ingestion_jobs import IngestionJobStore


class IngestionJobStoreTests(unittest.TestCase):
    def test_create_update_and_public_projection_are_atomic_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IngestionJobStore(Path(directory))
            record = store.create(
                file_name="paper.pdf",
                source_path=Path(directory) / "tmp.pdf",
                target_path=Path(directory) / "paper.pdf",
                replace_on_success=True,
                max_attempts=3,
            )
            updated = store.update(record["job_id"], status="running", attempts=1)
            public = store.public(updated)

            self.assertEqual(public["status"], "running")
            self.assertEqual(public["attempts"], 1)
            self.assertNotIn("_source_path", public)
            self.assertEqual(store.get(record["job_id"])["job_id"], record["job_id"])

    def test_list_is_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IngestionJobStore(Path(directory))
            first = store.create(
                file_name="first.pdf",
                source_path=Path(directory) / "first.tmp",
                target_path=Path(directory) / "first.pdf",
                replace_on_success=True,
                max_attempts=1,
            )
            second = store.create(
                file_name="second.pdf",
                source_path=Path(directory) / "second.tmp",
                target_path=Path(directory) / "second.pdf",
                replace_on_success=True,
                max_attempts=1,
            )
            listed = store.list()
            self.assertEqual({listed[0]["job_id"], listed[1]["job_id"]}, {first["job_id"], second["job_id"]})

    def test_worker_retries_and_persists_success(self):
        with tempfile.TemporaryDirectory() as directory:
            store = IngestionJobStore(Path(directory))
            record = store.create(
                file_name="paper.pdf",
                source_path=Path(directory) / "paper.tmp",
                target_path=Path(directory) / "paper.pdf",
                replace_on_success=True,
                max_attempts=2,
            )
            with patch.object(ingestion_jobs, "_store", store), patch.object(
                ingestion_jobs,
                "_process_job",
                side_effect=[RuntimeError("temporary qdrant outage"), {"chunks": 4, "document_id": "doc-1", "content_hash": "hash-1"}],
            ):
                ingestion_jobs._run_job(record["job_id"])

            result = store.get(record["job_id"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["chunks"], 4)
            self.assertEqual(result["document_id"], "doc-1")


if __name__ == "__main__":
    unittest.main()
