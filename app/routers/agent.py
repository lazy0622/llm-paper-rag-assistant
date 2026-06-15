from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.agent.research_agent import get_research_run, run_research_agent
from app.schemas import (
    EvidenceItem,
    ResearchRequest,
    ResearchResponse,
    ResearchRunResponse,
    ResearchRunSummary,
    SourceChunk,
    ToolCallLog,
)
from app.services.agent_run_store import list_research_runs, load_research_report

router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    try:
        result = run_research_agent(request.task, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research agent failed: {exc}") from exc
    return _to_research_response(result)


@router.get("/runs", response_model=list[ResearchRunSummary])
def list_runs() -> list[ResearchRunSummary]:
    # History only returns a compact index. Full traces and reports are fetched
    # by run_id to avoid sending large source chunks on every page load.
    return [ResearchRunSummary(**run) for run in list_research_runs()]


@router.get("/runs/{run_id}", response_model=ResearchRunResponse)
def get_run(run_id: str) -> ResearchRunResponse:
    result = get_research_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research run not found.")
    response = _to_research_response(result)
    return ResearchRunResponse(task=result["task"], **response.model_dump())


@router.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def get_run_report(run_id: str) -> PlainTextResponse:
    # Serve Markdown directly so Streamlit can offer a download button and the
    # same endpoint can be opened from Swagger during an interview demo.
    report = load_research_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Research report not found.")
    return PlainTextResponse(report, media_type="text/markdown; charset=utf-8")


def _to_research_response(result: dict) -> ResearchResponse:
    # Keep API serialization in one place because in-memory runs and persisted
    # JSON runs share the same response shape.
    return ResearchResponse(
        run_id=result["run_id"],
        plan=result["plan"],
        tool_calls=[ToolCallLog(**tool_call) for tool_call in result["tool_calls"]],
        final_report=result["final_report"],
        sources=[
            SourceChunk(
                file_name=source.get("file_name", "unknown"),
                page=source.get("page"),
                chunk_id=source.get("chunk_id"),
                content=source.get("content", ""),
                score=source.get("score"),
                evidence_level=source.get("evidence_level"),
                evidence_reason=source.get("evidence_reason"),
            )
            for source in result["sources"]
        ],
        evidence_items=[EvidenceItem(**item) for item in result.get("evidence_items", [])],
        overclaim_warnings=result.get("overclaim_warnings", []),
        created_at=result.get("created_at"),
        report_path=result.get("report_path"),
    )
