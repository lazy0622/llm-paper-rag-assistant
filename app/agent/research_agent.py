import time
import uuid
from typing import TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from app.services.llm_client import chat_completion
from app.services.evidence import (
    annotate_sources,
    append_overclaim_section,
    check_overclaim,
    format_evidence_map,
    grade_claim_evidence,
    split_claims,
)
from app.services.agent_run_store import load_research_run, save_research_run
from app.services.vector_store import list_indexed_documents, search_chunks


SUMMARY_PROMPT = PromptTemplate.from_template(
    """你是论文阅读助手。请只根据【检索片段】总结主题“{topic}”。

【检索片段】
{context}

【输出要求】
1. 用中文总结核心思想。
2. 提炼解决的问题、主要方法、优点和局限。
3. 必须提到依据来自哪些文件和页码。
4. 如果资料不足，明确写“资料不足”。
"""
)

COMPARE_PROMPT = PromptTemplate.from_template(
    """你是大模型应用岗位面试中的论文调研助手。

请根据以下单篇摘要，对多个方法做对比。

【调研任务】
{task}

【单篇摘要】
{summaries}

【输出要求】
1. 给出 Markdown 对比表，列包括：方法、核心思想、解决问题、优点、局限、适用场景。
2. 表格后总结 3 条工程启发。
3. 不要编造摘要中没有的论文结论。
"""
)

REPORT_PROMPT = PromptTemplate.from_template(
    """请把以下调研内容整理成一份可放入项目 README 或面试展示的 Markdown 报告。

【调研任务】
{task}

【方法对比】
{comparison}

【证据地图】
{evidence_map}

【检索质量记录】
{retrieval_eval}

【输出要求】
1. 标题清晰。
2. 按“调研目标、为什么重要、技术缺口、方法路线、证据地图、方法对比、工程启发、局限与下一步”组织。
3. 每个关键结论尽量带证据等级、文件名和页码；资料不足时明确标注“资料不足”。
4. 语言务实，不夸大为生产系统，不要把弱证据写成强结论。
"""
)


class ResearchState(TypedDict):
    run_id: str
    task: str
    top_k: int
    plan: list[str]
    documents: list[dict]
    topics: list[str]
    claims: list[str]
    topic_chunks: dict[str, list[dict]]
    claim_chunks: dict[str, list[dict]]
    paper_summaries: dict[str, str]
    comparison: str
    retrieval_eval: list[dict]
    evidence_items: list[dict]
    overclaim_warnings: list[str]
    final_report: str
    sources: list[dict]
    tool_calls: list[dict]
    created_at: str
    report_path: str


RUN_STORE: dict[str, dict] = {}


def run_research_agent(task: str, top_k: int = 5) -> dict:
    """Run a bounded Agentic RAG workflow and keep an inspectable run log."""
    run_id = str(uuid.uuid4())
    initial_state: ResearchState = {
        "run_id": run_id,
        "task": task,
        "top_k": top_k,
        "plan": [],
        "documents": [],
        "topics": [],
        "claims": [],
        "topic_chunks": {},
        "claim_chunks": {},
        "paper_summaries": {},
        "comparison": "",
        "retrieval_eval": [],
        "evidence_items": [],
        "overclaim_warnings": [],
        "final_report": "",
        "sources": [],
        "tool_calls": [],
        "created_at": "",
        "report_path": "",
    }
    result = _build_graph().invoke(initial_state)
    # Persist after the graph finishes so every tool call, evidence item, source,
    # and overclaim warning can be replayed even if the API process restarts.
    result = save_research_run(result)
    RUN_STORE[run_id] = result
    return result


def get_research_run(run_id: str) -> dict | None:
    result = RUN_STORE.get(run_id)
    if result is not None:
        return result

    # In-memory run logs are convenient during one process lifetime, but interview
    # demos need reproducibility after restart, so we fall back to the JSON file.
    result = load_research_run(run_id)
    if result is not None:
        RUN_STORE[run_id] = result
    return result


def _build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("plan_task", _plan_task)
    graph.add_node("list_documents", _list_documents)
    graph.add_node("segment_claims", _segment_claims)
    graph.add_node("search_chunks", _search_research_chunks)
    graph.add_node("grade_evidence", _grade_evidence)
    graph.add_node("summarize_paper", _summarize_topics)
    graph.add_node("compare_methods", _compare_methods)
    graph.add_node("write_report", _write_report)
    graph.add_node("check_overclaim", _check_overclaim)

    graph.set_entry_point("plan_task")
    graph.add_edge("plan_task", "list_documents")
    graph.add_edge("list_documents", "segment_claims")
    graph.add_edge("segment_claims", "search_chunks")
    graph.add_edge("search_chunks", "grade_evidence")
    graph.add_edge("grade_evidence", "summarize_paper")
    graph.add_edge("summarize_paper", "compare_methods")
    graph.add_edge("compare_methods", "write_report")
    graph.add_edge("write_report", "check_overclaim")
    graph.add_edge("check_overclaim", END)
    return graph.compile()


def _plan_task(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    topics = _extract_topics(state["task"])
    plan = [
        "识别调研主题和候选论文方向",
        "读取知识库文件清单",
        "拆分可验证 claim",
        "按 claim 检索论文片段",
        "给检索证据分级",
        "生成单 claim/主题摘要",
        "对比方法差异并生成 Nature 式调研报告",
        "检查过度声称和资料不足边界",
    ]
    state["topics"] = topics
    state["plan"] = plan
    _append_tool_call(state, "plan_task", state["task"], f"topics={topics}", start)
    return state


def _list_documents(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    documents = list_indexed_documents()
    state["documents"] = documents
    output = "; ".join(f"{document['file_name']}({document['chunks']} chunks)" for document in documents)
    _append_tool_call(state, "list_documents", "current knowledge base", output, start)
    return state


def _segment_claims(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    claims = split_claims(state["task"], state["topics"])
    state["claims"] = claims
    # 把调研任务拆成可验证 claim，是借鉴 Nature citation 的核心动作：
    # 先明确“要证明什么”，再去检索证据，避免整段任务被模型自由发挥。
    _append_tool_call(state, "segment_claims", state["task"], "\n".join(claims), start)
    return state


def _search_research_chunks(state: ResearchState) -> ResearchState:
    claims = state["claims"] or state["topics"]
    for claim in claims:
        start = time.perf_counter()
        query = _claim_query(claim)
        chunks = search_chunks(query, state["top_k"])
        annotated_chunks = annotate_sources(chunks, claim)
        state["claim_chunks"][claim] = annotated_chunks
        state["sources"].extend(annotated_chunks)
        state["retrieval_eval"].append(
            {
                "topic": claim,
                "query": query,
                "source_count": len(chunks),
                "top_score": chunks[0].get("score") if chunks else None,
                "source_files": sorted({chunk.get("file_name", "") for chunk in chunks if chunk.get("file_name")}),
            }
        )
        output = _format_sources(chunks) if chunks else "资料不足：未检索到可用片段"
        _append_tool_call(state, "search_chunks", query, output, start, source_count=len(chunks))
    return state


def _grade_evidence(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    evidence_items = [
        grade_claim_evidence(claim, state["claim_chunks"].get(claim, []))
        for claim in state["claims"]
    ]
    state["evidence_items"] = evidence_items
    _append_tool_call(
        state,
        "grade_evidence",
        "\n".join(state["claims"]),
        format_evidence_map(evidence_items),
        start,
        source_count=sum(item["source_count"] for item in evidence_items),
    )
    return state


def _summarize_topics(state: ResearchState) -> ResearchState:
    for topic, chunks in state["claim_chunks"].items():
        start = time.perf_counter()
        if not chunks:
            summary = f"{topic}: 资料不足，未检索到可用片段。"
        else:
            prompt = SUMMARY_PROMPT.format(topic=topic, context=_format_context(chunks))
            summary = chat_completion(prompt)
        state["paper_summaries"][topic] = summary
        _append_tool_call(state, "summarize_paper", topic, summary, start, source_count=len(chunks))
    return state


def _compare_methods(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    summaries = "\n\n".join(f"## {topic}\n{summary}" for topic, summary in state["paper_summaries"].items())
    prompt = COMPARE_PROMPT.format(task=state["task"], summaries=summaries)
    state["comparison"] = chat_completion(prompt)
    _append_tool_call(state, "compare_methods", state["task"], state["comparison"], start)
    return state


def _write_report(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    prompt = REPORT_PROMPT.format(
        task=state["task"],
        comparison=state["comparison"],
        evidence_map=format_evidence_map(state["evidence_items"]),
        retrieval_eval=_format_retrieval_eval(state["retrieval_eval"]),
    )
    state["final_report"] = chat_completion(prompt)
    _append_tool_call(state, "write_report", state["task"], state["final_report"], start)
    state["sources"] = _deduplicate_sources(state["sources"])
    return state


def _check_overclaim(state: ResearchState) -> ResearchState:
    start = time.perf_counter()
    warnings = check_overclaim(state["final_report"], state["evidence_items"])
    state["overclaim_warnings"] = warnings
    state["final_report"] = append_overclaim_section(state["final_report"], warnings)
    output = "\n".join(warnings) if warnings else "未发现明显过度声称风险。"
    _append_tool_call(state, "check_overclaim", state["task"], output, start)
    return state


def _extract_topics(task: str) -> list[str]:
    topic_map = {
        "Self-RAG": ["self-rag", "self rag", "自反思"],
        "CRAG": ["crag", "corrective rag", "corrective-rag"],
        "GraphRAG": ["graphrag", "graph rag", "图 rag"],
        "RAG": ["rag", "检索增强"],
        "ReAct": ["react", "tool use", "agent"],
    }
    lowered = task.lower()
    topics = [topic for topic, aliases in topic_map.items() if any(alias in lowered for alias in aliases)]
    specific_rag_topics = {"Self-RAG", "CRAG", "GraphRAG"}
    if "RAG" in topics and specific_rag_topics & set(topics):
        standalone_rag_markers = ["普通rag", "基础rag", "standard rag", "basic rag", "rag 原论文"]
        if not any(marker in lowered for marker in standalone_rag_markers):
            topics.remove("RAG")
    return topics or ["RAG", "Self-RAG", "CRAG", "GraphRAG"]


def _claim_query(claim: str) -> str:
    for topic in ["Self-RAG", "CRAG", "GraphRAG", "ReAct", "RAG"]:
        if topic.lower() in claim.lower():
            return _topic_query(topic)
    return claim


def _topic_query(topic: str) -> str:
    query_map = {
        "Self-RAG": "Self-RAG retrieve generate critique self reflection factuality citation accuracy",
        "CRAG": "Corrective Retrieval Augmented Generation retrieval evaluator decompose recompose",
        "GraphRAG": "GraphRAG local to global query focused summarization community summaries",
        "RAG": "Retrieval-Augmented Generation parametric non-parametric memory knowledge intensive NLP",
        "ReAct": "ReAct reasoning acting language models tool use actions observations",
    }
    return query_map.get(topic, topic)


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        page = chunk.get("page")
        page_text = f"p.{page}" if page else "page unknown"
        parts.append(
            f"[{index}] {chunk.get('file_name', 'unknown')} {page_text} "
            f"score={chunk.get('score')}\n{chunk.get('content', '')}"
        )
    return "\n\n".join(parts)


def _format_sources(chunks: list[dict]) -> str:
    return "; ".join(
        f"{chunk.get('file_name', 'unknown')} p.{chunk.get('page')} score={chunk.get('score', 0):.4f}"
        for chunk in chunks[:5]
    )


def _format_retrieval_eval(rows: list[dict]) -> str:
    return "\n".join(
        f"- {row['topic']}: query={row['query']}, source_count={row['source_count']}, "
        f"top_score={row['top_score']}, files={row['source_files']}"
        for row in rows
    )


def _deduplicate_sources(sources: list[dict]) -> list[dict]:
    deduplicated = {}
    for source in sources:
        key = source.get("chunk_id") or f"{source.get('file_name')}:{source.get('page')}:{source.get('content', '')[:40]}"
        deduplicated[key] = source
    return list(deduplicated.values())


def _append_tool_call(
    state: ResearchState,
    tool_name: str,
    tool_input: str,
    tool_output: str,
    start: float,
    source_count: int = 0,
    status: str = "success",
) -> None:
    state["tool_calls"].append(
        {
            "tool_name": tool_name,
            "tool_input": str(tool_input)[:800],
            "tool_output": str(tool_output)[:1600],
            "status": status,
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "source_count": source_count,
        }
    )
