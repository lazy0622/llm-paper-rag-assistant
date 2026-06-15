import json
import time
from typing import Any

from app.config import settings


def now_ms() -> int:
    return int(time.time() * 1000)


def log_rag_chat(
    *,
    run_id: str,
    question: str,
    answer_mode: str,
    top_k: int,
    source_count: int,
    top_score: float | None,
    duration_ms: int,
    status: str,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist one RAG chat trace for demo review and gateway log correlation."""
    settings.rag_log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_ms": now_ms(),
        "run_id": run_id,
        "question": question,
        "answer_mode": answer_mode,
        "top_k": top_k,
        "source_count": source_count,
        "top_score": top_score,
        "duration_ms": duration_ms,
        "status": status,
        "error": error,
        "extra": extra or {},
    }
    with settings.rag_log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_recent_rag_logs(limit: int = 50) -> list[dict[str, Any]]:
    if not settings.rag_log_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in settings.rag_log_path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
