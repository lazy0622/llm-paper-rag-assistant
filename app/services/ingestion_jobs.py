"""Durable local document-ingestion jobs with retries and restart recovery."""

from __future__ import annotations

import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings


class IngestionJobStore:
    """Small atomic JSON store; replaceable by Redis/PostgreSQL later."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._lock = threading.RLock()

    def create(
        self,
        *,
        file_name: str,
        source_path: Path,
        target_path: Path,
        replace_on_success: bool,
        max_attempts: int,
    ) -> dict[str, Any]:
        with self._lock:
            now = _now()
            record = {
                "job_id": f"ingest-{uuid.uuid4().hex}",
                "file_name": file_name,
                "status": "queued",
                "attempts": 0,
                "max_attempts": max(1, max_attempts),
                "chunks": 0,
                "document_id": None,
                "content_hash": None,
                "error": None,
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "_source_path": str(source_path),
                "_target_path": str(target_path),
                "_replace_on_success": replace_on_success,
            }
            self._write(record)
            return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        self.directory.mkdir(parents=True, exist_ok=True)
        records = []
        with self._lock:
            for path in self.directory.glob("*.json"):
                try:
                    records.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
        records.sort(key=lambda record: record.get("created_at", ""), reverse=True)
        return records[: max(1, limit)]

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self.get(job_id)
            if record is None:
                raise KeyError(f"Unknown ingestion job: {job_id}")
            record.update(fields)
            self._write(record)
            return record

    def public(self, record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if not key.startswith("_")}

    def _write(self, record: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self._path(record["job_id"])
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def _path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"


_store = IngestionJobStore(settings.ingestion_job_dir)
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def enqueue_document_ingestion(
    *,
    file_name: str,
    source_path: Path,
    target_path: Path,
    replace_on_success: bool,
) -> dict[str, Any]:
    record = _store.create(
        file_name=file_name,
        source_path=source_path,
        target_path=target_path,
        replace_on_success=replace_on_success,
        max_attempts=settings.ingestion_max_attempts,
    )
    _executor_instance().submit(_run_job, record["job_id"])
    return _store.public(record)


def get_ingestion_job(job_id: str) -> dict[str, Any] | None:
    record = _store.get(job_id)
    return _store.public(record) if record else None


def list_ingestion_jobs(limit: int = 50) -> list[dict[str, Any]]:
    return [_store.public(record) for record in _store.list(limit=limit)]


def recover_pending_ingestion_jobs() -> int:
    """Resume jobs left in flight after an API process restart."""
    recovered = 0
    for record in _store.list(limit=10000):
        if record.get("status") not in {"queued", "running", "retrying"}:
            continue
        source_path = Path(record.get("_source_path", ""))
        if not source_path.exists():
            _store.update(
                record["job_id"],
                status="failed",
                error="Ingestion source file is missing after process restart.",
                finished_at=_now(),
            )
            continue
        # A process may have died after marking an attempt running. Reopen
        # that attempt instead of incrementing past max_attempts on recovery.
        if record.get("status") == "running":
            record = _store.update(record["job_id"], status="queued", attempts=max(0, int(record.get("attempts", 0)) - 1))
        _executor_instance().submit(_run_job, record["job_id"])
        recovered += 1
    return recovered


def shutdown_ingestion_executor() -> None:
    global _executor
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False, cancel_futures=False)
            _executor = None


def _run_job(job_id: str) -> None:
    record = _store.get(job_id)
    if record is None:
        return

    start_attempt = int(record.get("attempts", 0)) + 1
    max_attempts = int(record.get("max_attempts", settings.ingestion_max_attempts))
    for attempt in range(start_attempt, max_attempts + 1):
        _store.update(job_id, status="running", attempts=attempt, started_at=_now(), error=None)
        try:
            result = _process_job(record)
        except Exception as exc:  # noqa: BLE001 - persisted for operator diagnosis
            message = f"{type(exc).__name__}: {exc}"
            if attempt < max_attempts:
                _store.update(job_id, status="retrying", error=message, finished_at=None)
                continue
            _cleanup_temporary_source(record)
            _store.update(job_id, status="failed", error=message, finished_at=_now())
            return

        _store.update(
            job_id,
            status="succeeded",
            chunks=result["chunks"],
            document_id=result.get("document_id"),
            content_hash=result.get("content_hash"),
            error=None,
            finished_at=_now(),
        )
        return


def _process_job(record: dict[str, Any]) -> dict[str, Any]:
    from app.services.document_loader import load_document
    from app.services.splitter import split_pages
    from app.services.vector_store import prune_stale_document_versions, upsert_chunks

    source_path = Path(record["_source_path"])
    target_path = Path(record["_target_path"])
    pages = load_document(source_path, file_name=record["file_name"])
    chunks = split_pages(pages)
    stored = upsert_chunks(chunks)
    if stored == 0:
        raise ValueError("No valid text chunks were extracted from the document.")

    if record.get("_replace_on_success") and source_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
    elif source_path != target_path:
        source_path.unlink(missing_ok=True)

    document_id = chunks[0].get("document_id")
    index_version = chunks[0].get("index_version")
    if document_id and index_version:
        prune_stale_document_versions(record["file_name"], document_id, index_version)
    return {
        "chunks": stored,
        "document_id": document_id,
        "content_hash": chunks[0].get("content_hash"),
    }


def _cleanup_temporary_source(record: dict[str, Any]) -> None:
    source_path = Path(record.get("_source_path", ""))
    target_path = Path(record.get("_target_path", ""))
    if source_path != target_path:
        source_path.unlink(missing_ok=True)


def _executor_instance() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=max(1, settings.ingestion_workers),
                thread_name_prefix="rag-ingest",
            )
        return _executor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
