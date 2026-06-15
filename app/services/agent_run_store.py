import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT


RUN_DIR = PROJECT_ROOT / "reports" / "agent_runs"


def save_research_run(result: dict[str, Any]) -> dict[str, Any]:
    """Persist an Agent run for later review and interview demos.

    JSON keeps the full machine-readable trace, while Markdown is the artifact a
    reviewer or interviewer can open directly without running the backend.
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_id = result["run_id"]
    created_at = result.get("created_at") or datetime.now().isoformat(timespec="seconds")
    json_path = RUN_DIR / f"{run_id}.json"
    markdown_path = RUN_DIR / f"{run_id}.md"

    persisted = {
        **result,
        "created_at": created_at,
        "report_path": str(markdown_path),
    }
    json_path.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_markdown_report(persisted), encoding="utf-8")
    return persisted


def load_research_run(run_id: str) -> dict[str, Any] | None:
    """Load a persisted run after FastAPI restarts or in-memory state is lost."""
    json_path = RUN_DIR / f"{run_id}.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def list_research_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Return lightweight run summaries for history pages.

    The history UI should stay fast, so it reads only metadata instead of sending
    large tool outputs, sources, and final reports for every run.
    """
    if not RUN_DIR.exists():
        return []

    run_files = sorted(RUN_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    summaries = []
    for path in run_files[:limit]:
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summaries.append(
            {
                "run_id": run.get("run_id", path.stem),
                "task": run.get("task", ""),
                "created_at": run.get("created_at", ""),
                "report_path": run.get("report_path", str(RUN_DIR / f"{path.stem}.md")),
                "evidence_count": len(run.get("evidence_items", [])),
                "warning_count": len(run.get("overclaim_warnings", [])),
            }
        )
    return summaries


def load_research_report(run_id: str) -> str | None:
    """Read the Markdown artifact used for download and interview review."""
    markdown_path = RUN_DIR / f"{run_id}.md"
    if not markdown_path.exists():
        return None
    return markdown_path.read_text(encoding="utf-8")


def _build_markdown_report(run: dict[str, Any]) -> str:
    """Build a self-contained report that can be opened without the web app."""
    evidence_items = run.get("evidence_items", [])
    warnings = run.get("overclaim_warnings", [])
    sources = run.get("sources", [])

    parts = [
        "# Agentic Paper Research Report",
        "",
        "## Run Metadata",
        f"- run_id: `{run.get('run_id', '')}`",
        f"- created_at: `{run.get('created_at', '')}`",
        f"- task: {run.get('task', '')}",
        f"- evidence_items: {len(evidence_items)}",
        f"- overclaim_warnings: {len(warnings)}",
        f"- sources: {len(sources)}",
        "",
        "## Evidence Overview",
        _format_evidence_table(evidence_items),
        "",
        "## Overclaim Warnings",
        _format_warning_list(warnings),
        "",
        "## Final Report",
        run.get("final_report", ""),
        "",
        "## Tool Calls",
        _format_tool_calls(run.get("tool_calls", [])),
        "",
        "## Sources",
        _format_sources(sources),
        "",
    ]
    return "\n".join(parts)


def _format_evidence_table(evidence_items: list[dict[str, Any]]) -> str:
    if not evidence_items:
        return "No evidence items."

    lines = [
        "| Claim | Evidence | Source Count | Top Score | Files | Pages |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in evidence_items:
        files = "<br>".join(_escape_table_text(file) for file in item.get("source_files", [])) or "-"
        pages = ", ".join(str(page) for page in item.get("source_pages", [])) or "-"
        top_score = item.get("top_score")
        score_text = "-" if top_score is None else f"{float(top_score):.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_text(item.get("claim", "")),
                    _escape_table_text(item.get("evidence_level", "")),
                    str(item.get("source_count", 0)),
                    score_text,
                    files,
                    pages,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _format_warning_list(warnings: list[str]) -> str:
    if not warnings:
        return "- No obvious overclaim risk detected."
    return "\n".join(f"- {warning}" for warning in warnings)


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    if not tool_calls:
        return "No tool calls."
    lines = []
    for index, call in enumerate(tool_calls, start=1):
        lines.extend(
            [
                f"### {index}. {call.get('tool_name', '')}",
                f"- status: `{call.get('status', '')}`",
                f"- duration_ms: `{call.get('duration_ms', '')}`",
                f"- source_count: `{call.get('source_count', 0)}`",
                "",
                "**Input**",
                "",
                "```text",
                str(call.get("tool_input", "")),
                "```",
                "",
                "**Output**",
                "",
                "```text",
                str(call.get("tool_output", "")),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "No sources."
    lines = []
    for index, source in enumerate(sources, start=1):
        lines.extend(
            [
                f"### Source {index}",
                f"- file: `{source.get('file_name', 'unknown')}`",
                f"- page: `{source.get('page', 'unknown')}`",
                f"- chunk_id: `{source.get('chunk_id', '')}`",
                f"- score: `{source.get('score', '')}`",
                f"- evidence_level: `{source.get('evidence_level', '')}`",
                f"- evidence_reason: {source.get('evidence_reason', '')}",
                "",
                "```text",
                str(source.get("content", ""))[:1200],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _escape_table_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")
