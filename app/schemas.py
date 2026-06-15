from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    allow_fallback: bool = True
    answer_style: Literal["brief", "standard", "detailed"] = "standard"


class SourceChunk(BaseModel):
    file_name: str
    page: int | None = None
    chunk_id: str | None = None
    content: str
    score: float | None = None
    rerank_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    final_score: float | None = None
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    retrieval_source: str | None = None
    evidence_level: Literal[
        "strong_support",
        "partial_support",
        "background_support",
        "weak_support",
        "insufficient",
    ] | None = None
    evidence_reason: str | None = None


class ChatResponse(BaseModel):
    run_id: str
    answer: str
    answer_mode: Literal["grounded", "fallback", "no_answer", "metadata"]
    rewritten_query: str | None = None
    sources: list[SourceChunk] = Field(default_factory=list)


class IngestResponse(BaseModel):
    file_name: str
    chunks: int
    status: str


class DocumentSummary(BaseModel):
    file_name: str
    chunks: int
    pages: list[int] = Field(default_factory=list)


class DocumentActionResponse(BaseModel):
    file_name: str
    affected_chunks: int
    status: str


class EvidenceItem(BaseModel):
    claim: str
    evidence_level: Literal[
        "strong_support",
        "partial_support",
        "background_support",
        "weak_support",
        "insufficient",
    ]
    evidence_reason: str
    source_files: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list)
    source_count: int = 0
    top_score: float | None = None


class ResearchRequest(BaseModel):
    task: str = Field(..., min_length=1)
    top_k: int = 5


class ToolCallLog(BaseModel):
    tool_name: str
    tool_input: str
    tool_output: str
    status: Literal["success", "failed"]
    duration_ms: int
    source_count: int = 0


class ResearchResponse(BaseModel):
    run_id: str
    plan: list[str]
    tool_calls: list[ToolCallLog]
    final_report: str
    sources: list[SourceChunk] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    overclaim_warnings: list[str] = Field(default_factory=list)
    created_at: str | None = None
    report_path: str | None = None


class ResearchRunResponse(ResearchResponse):
    task: str


class ResearchRunSummary(BaseModel):
    run_id: str
    task: str
    created_at: str | None = None
    report_path: str | None = None
    evidence_count: int = 0
    warning_count: int = 0
