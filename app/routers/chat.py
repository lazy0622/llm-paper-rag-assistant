import uuid

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.evidence import annotate_sources
from app.services.query_rewriter import rewrite_query_for_retrieval
from app.services.rag_chain import (
    answer_with_context,
    answer_with_fallback,
    is_metadata_question,
    is_strict_knowledge_question,
    needs_fallback,
)
from app.services.rag_request_logger import log_rag_chat, now_ms, read_recent_rag_logs
from app.services.vector_store import list_indexed_documents, search_chunks

router = APIRouter()


NO_ANSWER_TEXT = "资料中没有明确依据。当前已关闭本地模型补充回答，因此不使用模型通用知识补充。"


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer one question with metadata, grounded RAG, fallback, or no_answer mode."""
    started = now_ms()
    run_id = f"rag-{uuid.uuid4().hex}"
    top_k = request.top_k or settings.top_k
    chunks: list[dict] = []
    answer_mode = "grounded"
    rewritten_query: str | None = None

    try:
        if is_metadata_question(request.question):
            answer = _format_document_inventory()
            log_rag_chat(
                run_id=run_id,
                question=request.question,
                answer_mode="metadata",
                top_k=top_k,
                source_count=0,
                top_score=None,
                duration_ms=now_ms() - started,
                status="success",
            )
            return ChatResponse(run_id=run_id, answer=answer, answer_mode="metadata", rewritten_query=None, sources=[])

        rewritten_query = rewrite_query_for_retrieval(request.question, run_id=run_id)
        chunks = search_chunks(rewritten_query, top_k, run_id=run_id)
        effective_allow_fallback = request.allow_fallback and not is_strict_knowledge_question(request.question)

        if not chunks:
            answer = (
                answer_with_fallback(request.question, request.answer_style, run_id=run_id)
                if effective_allow_fallback
                else NO_ANSWER_TEXT
            )
            answer_mode = "fallback" if effective_allow_fallback else "no_answer"
        else:
            # The rewritten query is only for retrieval. The final answer still
            # uses the user's original question to avoid semantic drift.
            answer = answer_with_context(request.question, chunks, request.answer_style, run_id=run_id)
            if needs_fallback(answer):
                if effective_allow_fallback:
                    answer = answer_with_fallback(request.question, request.answer_style, run_id=run_id)
                    answer_mode = "fallback"
                else:
                    answer = NO_ANSWER_TEXT
                    answer_mode = "no_answer"
    except Exception as exc:
        log_rag_chat(
            run_id=run_id,
            question=request.question,
            answer_mode=answer_mode,
            top_k=top_k,
            source_count=0,
            top_score=None,
            duration_ms=now_ms() - started,
            status="failed",
            error=str(exc),
            extra={"rewritten_query": rewritten_query},
        )
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    source_chunks = annotate_sources(chunks, request.question) if answer_mode == "grounded" else []
    sources = [_to_source_chunk(chunk) for chunk in source_chunks]
    top_score = source_chunks[0].get("score") if source_chunks else None
    log_rag_chat(
        run_id=run_id,
        question=request.question,
        answer_mode=answer_mode,
        top_k=top_k,
        source_count=len(source_chunks),
        top_score=top_score,
        duration_ms=now_ms() - started,
        status="success",
        extra={
            "allow_fallback": request.allow_fallback,
            "answer_style": request.answer_style,
            "rewritten_query": rewritten_query,
        },
    )
    return ChatResponse(
        run_id=run_id,
        answer=answer,
        answer_mode=answer_mode,
        rewritten_query=rewritten_query,
        sources=sources,
    )


@router.get("/logs")
def chat_logs(limit: int = 50) -> list[dict]:
    return read_recent_rag_logs(limit=limit)


def _to_source_chunk(chunk: dict) -> SourceChunk:
    return SourceChunk(
        file_name=chunk["file_name"],
        page=chunk.get("page"),
        chunk_id=chunk.get("chunk_id"),
        content=chunk.get("content", ""),
        score=chunk.get("score"),
        rerank_score=chunk.get("rerank_score"),
        keyword_score=chunk.get("keyword_score"),
        rrf_score=chunk.get("rrf_score"),
        final_score=chunk.get("final_score"),
        semantic_rank=chunk.get("semantic_rank"),
        keyword_rank=chunk.get("keyword_rank"),
        retrieval_source=chunk.get("retrieval_source"),
        evidence_level=chunk.get("evidence_level"),
        evidence_reason=chunk.get("evidence_reason"),
    )


def _format_document_inventory() -> str:
    documents = list_indexed_documents()
    if not documents:
        return "当前知识库还没有入库文档。"

    lines = ["当前知识库已入库以下文档："]
    for index, document in enumerate(documents, start=1):
        pages = document.get("pages", [])
        page_text = f"{len(pages)} 页" if pages else "页码未知"
        lines.append(f"{index}. {document['file_name']}：{document['chunks']} 个 chunks，{page_text}")
    lines.append("\n这是知识库元信息查询结果，不是基于语义检索生成的论文内容回答。")
    return "\n".join(lines)
