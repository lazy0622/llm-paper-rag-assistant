import csv
import io
import os

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def check_api_health() -> bool:
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def upload_document(uploaded_file) -> dict:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    with httpx.Client(timeout=300, trust_env=False) as client:
        response = client.post(f"{API_BASE_URL}/documents/upload", files=files)
    response.raise_for_status()
    return response.json()


def list_documents() -> list[dict]:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get(f"{API_BASE_URL}/documents")
    response.raise_for_status()
    return response.json()


def delete_document(file_name: str) -> dict:
    with httpx.Client(timeout=60, trust_env=False) as client:
        response = client.delete(f"{API_BASE_URL}/documents/{file_name}")
    response.raise_for_status()
    return response.json()


def reindex_document(file_name: str) -> dict:
    with httpx.Client(timeout=300, trust_env=False) as client:
        response = client.post(f"{API_BASE_URL}/documents/{file_name}/reindex")
    response.raise_for_status()
    return response.json()


def ask_question(question: str, top_k: int, allow_fallback: bool, answer_style: str) -> dict:
    payload = {
        "question": question,
        "top_k": top_k,
        "allow_fallback": allow_fallback,
        "answer_style": answer_style,
    }
    with httpx.Client(timeout=300, trust_env=False) as client:
        response = client.post(f"{API_BASE_URL}/chat", json=payload)
    response.raise_for_status()
    return response.json()


def list_rag_logs(limit: int = 20) -> list[dict]:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get(f"{API_BASE_URL}/chat/logs", params={"limit": limit})
    response.raise_for_status()
    return response.json()


def run_research_agent(task: str, top_k: int) -> dict:
    payload = {"task": task, "top_k": top_k}
    with httpx.Client(timeout=600, trust_env=False) as client:
        response = client.post(f"{API_BASE_URL}/agent/research", json=payload)
    response.raise_for_status()
    return response.json()


def list_research_runs() -> list[dict]:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get(f"{API_BASE_URL}/agent/runs")
    response.raise_for_status()
    return response.json()


def get_research_report(run_id: str) -> str:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.get(f"{API_BASE_URL}/agent/runs/{run_id}/report")
    response.raise_for_status()
    return response.text


def build_agent_task_from_question(question: str) -> str:
    return (
        "请基于知识库，对以下问题做论文调研，拆解核心 claim，检索证据，"
        "生成带证据等级和引用来源的 Markdown 报告：\n\n"
        f"{question}"
    )


def init_page() -> None:
    st.set_page_config(page_title="论文知识库 RAG 助手", page_icon="📄", layout="wide")
    st.title("论文知识库 RAG 助手")
    st.caption("面向论文阅读和实验室知识库场景，支持 RAG 问答、Agentic 调研、评测与 Bad Case 分析。")


def render_sidebar() -> tuple[int, bool, str]:
    with st.sidebar:
        st.header("系统状态")
        if check_api_health():
            st.success("FastAPI 后端已连接")
        else:
            st.error("FastAPI 后端未连接")
            st.code(".\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload", language="powershell")

        st.divider()
        st.header("文档入库")
        uploaded_file = st.file_uploader("上传 PDF / DOCX / Markdown / TXT", type=["pdf", "docx", "md", "txt"])
        if uploaded_file and st.button("上传并向量化入库", use_container_width=True):
            with st.spinner("正在解析、切分、向量化并写入 Qdrant..."):
                try:
                    result = upload_document(uploaded_file)
                    st.success(f"入库成功：{result['chunks']} 个 chunks")
                    st.json(result)
                except httpx.HTTPStatusError as exc:
                    st.error(f"入库失败：{exc.response.text}")
                except httpx.HTTPError as exc:
                    st.error(f"请求失败：{exc}")

        st.divider()
        top_k = st.slider("检索 TopK", min_value=1, max_value=10, value=5)
        allow_fallback = st.toggle("允许本地模型补充回答", value=True)
        style_label = st.selectbox("回答详细程度", ["标准", "简洁", "详细"], index=0)
        answer_style = {"简洁": "brief", "标准": "standard", "详细": "detailed"}[style_label]
        st.caption("Query Rewrite 只影响检索；最终回答仍基于用户原问题。")

        st.divider()
        st.header("当前知识库文件")
        _render_document_sidebar()

    return top_k, allow_fallback, answer_style


def _render_document_sidebar() -> None:
    try:
        documents = list_documents()
    except httpx.HTTPError as exc:
        st.caption(f"文件列表读取失败：{exc}")
        return

    if not documents:
        st.caption("暂无入库文档。")
        return

    for document in documents:
        file_name = document["file_name"]
        pages = document.get("pages", [])
        page_text = f"{len(pages)} 页" if pages else "页码未知"
        st.caption(f"{file_name} | {document['chunks']} chunks | {page_text}")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("重建索引", key=f"reindex_{file_name}", use_container_width=True):
                try:
                    result = reindex_document(file_name)
                    st.success(f"重建完成：{result['affected_chunks']} chunks")
                    st.rerun()
                except httpx.HTTPError as exc:
                    st.error(f"重建失败：{exc}")
        with col_b:
            if st.button("删除索引", key=f"delete_{file_name}", use_container_width=True):
                try:
                    result = delete_document(file_name)
                    st.warning(f"已删除：{result['affected_chunks']} chunks")
                    st.rerun()
                except httpx.HTTPError as exc:
                    st.error(f"删除失败：{exc}")


def render_rag_chat(top_k: int, allow_fallback: bool, answer_style: str) -> None:
    st.subheader("RAG 问答")
    st.caption("适合问单个明确问题；如果问题复杂，可以升级为 Agent 调研任务。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_rag_question" not in st.session_state:
        st.session_state.last_rag_question = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                st.markdown(message["content"])

    question = st.chat_input("请输入你的论文知识库问题")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                result = ask_question(question, top_k=top_k, allow_fallback=allow_fallback, answer_style=answer_style)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "run_id": result.get("run_id"),
                        "rewritten_query": result.get("rewritten_query"),
                        "answer_mode": result.get("answer_mode", "grounded"),
                        "sources": result.get("sources", []),
                    }
                )
                st.session_state.last_rag_question = question
                st.rerun()
            except httpx.HTTPError as exc:
                st.error(f"问答请求失败：{exc}")

    if st.session_state.last_rag_question:
        if st.button("升级为论文调研任务", use_container_width=True):
            st.session_state.agent_task_seed = build_agent_task_from_question(st.session_state.last_rag_question)
            st.success("已生成 Agent 调研任务，请切换到“论文调研 Agent”页运行。")


def _render_assistant_message(message: dict) -> None:
    answer_mode = message.get("answer_mode", "grounded")
    if answer_mode == "grounded":
        st.success("回答模式：基于知识库 sources 回答")
    elif answer_mode == "fallback":
        st.warning("回答模式：本地模型通用知识补充，答案不代表知识库来源")
    elif answer_mode == "metadata":
        st.info("回答模式：知识库元信息查询")
    elif answer_mode == "no_answer":
        st.error("回答模式：资料不足，严格拒答")

    if message.get("run_id"):
        st.caption(f"run_id: {message.get('run_id')}")
    if message.get("rewritten_query"):
        st.caption(f"检索改写 query: {message.get('rewritten_query')}")
    st.markdown(message.get("content", ""))
    _render_sources(message.get("sources", []), answer_mode)


def _render_sources(sources: list[dict], answer_mode: str) -> None:
    if not sources:
        if answer_mode == "fallback":
            st.info("补充回答不展示引用来源，因为答案来自本地模型通用知识，不来自知识库 sources。")
        elif answer_mode == "metadata":
            st.info("元信息查询来自 Qdrant payload 聚合，不展示语义检索引用来源。")
        elif answer_mode == "no_answer":
            st.info("严格拒答模式下没有可采用来源。")
        return

    st.subheader("引用来源")
    st.caption("sources 用于人工核验；score 是相似度，不等于事实证明。")
    for index, source in enumerate(sources, start=1):
        title = _build_source_title(index, source)
        with st.expander(title):
            st.caption(f"chunk_id: {source.get('chunk_id')}")
            _render_source_scores(source)
            if source.get("evidence_level"):
                st.info(
                    f"证据等级：{source.get('evidence_level')}。"
                    f"{source.get('evidence_reason') or 'score 是语义相似度，不等于事实证明。'}"
                )
            st.write(source.get("content", ""))


def _build_source_title(index: int, source: dict) -> str:
    title = f"{index}. {source.get('file_name', 'unknown')}"
    if source.get("page") is not None:
        title += f" | 第 {source.get('page')} 页"
    if source.get("score") is not None:
        title += f" | score={source.get('score'):.4f}"
    if source.get("rerank_score") is not None:
        title += f" | rerank={source.get('rerank_score'):.4f}"
    if source.get("final_score") is not None:
        title += f" | final={source.get('final_score'):.4f}"
    if source.get("retrieval_source"):
        title += f" | {source.get('retrieval_source')}"
    if source.get("evidence_level"):
        title += f" | evidence={source.get('evidence_level')}"
    return title


def _render_source_scores(source: dict) -> None:
    if source.get("keyword_score") is not None:
        st.caption(f"keyword_score={source.get('keyword_score'):.4f}")
    if source.get("rrf_score") is not None:
        st.caption(f"rrf_score={source.get('rrf_score'):.4f}")
    rank_parts = []
    if source.get("semantic_rank") is not None:
        rank_parts.append(f"semantic_rank={source.get('semantic_rank')}")
    if source.get("keyword_rank") is not None:
        rank_parts.append(f"keyword_rank={source.get('keyword_rank')}")
    if rank_parts:
        st.caption(" | ".join(rank_parts))


def render_agent_page(top_k: int) -> None:
    st.subheader("论文调研 Agent")
    st.caption("适合处理多论文对比、方法总结、证据分级和 Markdown 报告生成。")
    _render_research_history()

    default_task = st.session_state.get("agent_task_seed", "对比 Self-RAG、CRAG、GraphRAG 的核心思想、优缺点和适用场景")
    task = st.text_area("调研任务", value=default_task, height=140)

    if st.button("运行 Agent 调研", type="primary", use_container_width=True):
        with st.spinner("Agent 正在拆解任务、检索证据并生成报告..."):
            try:
                result = run_research_agent(task, top_k=top_k)
                st.session_state.last_agent_result = result
            except httpx.HTTPError as exc:
                st.error(f"Agent 请求失败：{exc}")

    if st.session_state.get("last_agent_result"):
        render_research_result(st.session_state.last_agent_result)


def render_research_result(result: dict) -> None:
    st.markdown("### 执行结果")
    st.caption(f"run_id: {result.get('run_id')}")
    if result.get("report_path"):
        st.caption(f"report_path: {result.get('report_path')}")

    if result.get("run_id"):
        try:
            report_text = get_research_report(result["run_id"])
            st.download_button(
                "下载 Markdown 报告",
                data=report_text,
                file_name=f"{result['run_id']}_research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except httpx.HTTPError:
            st.caption("Markdown 报告暂不可下载。")

    if result.get("plan"):
        st.markdown("### 任务计划")
        for item in result["plan"]:
            st.write(f"- {item}")

    _render_tool_calls(result.get("tool_calls", []))
    _render_evidence_items(result.get("evidence_items", []))

    st.markdown("### 最终调研报告")
    st.markdown(result.get("final_report", ""))

    warnings = result.get("overclaim_warnings", [])
    if warnings:
        st.markdown("### 过度声称风险")
        for warning in warnings:
            st.warning(warning)

    if result.get("sources"):
        st.markdown("### 报告引用来源")
        _render_sources(result.get("sources", []), "grounded")
    else:
        st.info("本次调研没有可展示的引用来源。")


def _render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    st.markdown("### 工具调用日志")
    for index, tool_call in enumerate(tool_calls, start=1):
        title = (
            f"{index}. {tool_call.get('tool_name')} | {tool_call.get('status')} | "
            f"{tool_call.get('duration_ms')} ms | sources={tool_call.get('source_count')}"
        )
        with st.expander(title):
            st.caption("input")
            st.code(tool_call.get("tool_input", ""))
            st.caption("output")
            st.write(tool_call.get("tool_output", ""))


def _render_evidence_items(evidence_items: list[dict]) -> None:
    if not evidence_items:
        return
    st.markdown("### 证据地图")
    st.caption("证据等级用于提醒：相关片段不一定能支撑强结论。")
    for index, item in enumerate(evidence_items, start=1):
        title = f"{index}. {item.get('evidence_level')} | sources={item.get('source_count')} | {item.get('claim')}"
        with st.expander(title):
            st.write(item.get("evidence_reason", ""))
            st.write("source_files:", ", ".join(item.get("source_files", [])) or "none")
            st.write("source_pages:", item.get("source_pages", []))
            st.write("top_score:", item.get("top_score"))


def render_rag_evaluation(top_k: int, answer_style: str) -> None:
    st.subheader("RAG 评测与 Bad Case 分析")
    st.caption("批量调用 /chat，记录 rewritten_query、source_hit、answer_mode、Hybrid/Rerank 分数和 bad_case_type。")

    uploaded_eval = st.file_uploader("上传 QA CSV", type=["csv"], key="eval_csv_upload")
    use_default = st.checkbox("使用项目内置 data/eval/qa_pairs.csv", value=True)

    if st.button("运行评测", type="primary", use_container_width=True):
        rows = _load_eval_rows(uploaded_eval, use_default)
        if not rows:
            st.warning("没有可评测的问题。")
            return
        results = []
        progress = st.progress(0)
        for index, row in enumerate(rows, start=1):
            question = row.get("question", "").strip()
            if not question:
                continue
            try:
                result = ask_question(question, top_k=top_k, allow_fallback=False, answer_style=answer_style)
                sources = result.get("sources", [])
                source_files = sorted({source.get("file_name", "") for source in sources if source.get("file_name")})
                expected_source = row.get("source_file", "")
                top_source = sources[0] if sources else {}
                top_score = top_source.get("score")
                source_hit = bool(expected_source and expected_source in source_files)
                bad_case_type = _classify_eval_bad_case(
                    answer_mode=result.get("answer_mode", ""),
                    source_count=len(sources),
                    source_hit=source_hit,
                    top_score=top_score,
                    expected_source=expected_source,
                )
                results.append(
                    {
                        "question": question,
                        "rewritten_query": result.get("rewritten_query", ""),
                        "expected_answer": row.get("expected_answer", ""),
                        "expected_source": expected_source,
                        "answer_mode": result.get("answer_mode", ""),
                        "source_count": len(sources),
                        "source_files": ";".join(source_files),
                        "source_hit": source_hit,
                        "top_score": _format_eval_score(top_score),
                        "top_rerank_score": _format_eval_score(top_source.get("rerank_score")),
                        "top_keyword_score": _format_eval_score(top_source.get("keyword_score")),
                        "top_final_score": _format_eval_score(top_source.get("final_score")),
                        "top_retrieval_source": top_source.get("retrieval_source", ""),
                        "bad_case_type": bad_case_type,
                        "run_id": result.get("run_id", ""),
                    }
                )
            except httpx.HTTPError as exc:
                results.append(
                    {
                        "question": question,
                        "rewritten_query": "",
                        "expected_answer": row.get("expected_answer", ""),
                        "expected_source": row.get("source_file", ""),
                        "answer_mode": "request_failed",
                        "source_count": 0,
                        "source_files": "",
                        "source_hit": False,
                        "top_score": "",
                        "top_rerank_score": "",
                        "top_keyword_score": "",
                        "top_final_score": "",
                        "top_retrieval_source": "",
                        "bad_case_type": f"request_failed: {exc}",
                        "run_id": "",
                    }
                )
            progress.progress(index / len(rows))

        _render_eval_results(results)
        _render_recent_rag_logs()


def _load_eval_rows(uploaded_eval, use_default: bool) -> list[dict]:
    if uploaded_eval is not None:
        text = uploaded_eval.getvalue().decode("utf-8-sig")
        return list(csv.DictReader(io.StringIO(text)))
    if use_default:
        path = os.path.join(os.path.dirname(__file__), "data", "eval", "qa_pairs.csv")
        with open(path, "r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    return []


def _classify_eval_bad_case(
    *,
    answer_mode: str,
    source_count: int,
    source_hit: bool,
    top_score: float | None,
    expected_source: str,
) -> str:
    if answer_mode == "fallback":
        return "fallback_triggered"
    if answer_mode == "no_answer":
        return "strict_no_answer"
    if source_count == 0:
        return "no_sources"
    if expected_source and not source_hit:
        return "expected_source_missed"
    if top_score is not None and top_score < 0.45:
        return "low_top_score"
    return "ok"


def _format_eval_score(score: float | None) -> str:
    return "" if score is None else f"{score:.4f}"


def _render_eval_results(results: list[dict]) -> None:
    total = len(results)
    source_hits = sum(1 for row in results if row["source_hit"])
    bad_cases = [row for row in results if row["bad_case_type"] != "ok"]
    grounded = sum(1 for row in results if row["answer_mode"] == "grounded")
    no_answer = sum(1 for row in results if row["answer_mode"] == "no_answer")
    hybrid = sum(1 for row in results if row.get("top_retrieval_source") == "hybrid")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("评测问题", total)
    col2.metric("命中来源", source_hits)
    col3.metric("grounded", grounded)
    col4.metric("no_answer", no_answer)

    st.caption(f"Hybrid Top 来源数量：{hybrid}。如果该值增加，说明关键词召回与向量召回发生了融合。")
    st.markdown("### 评测结果")
    st.dataframe(results, use_container_width=True, hide_index=True)

    if bad_cases:
        st.markdown("### 典型 Bad Case")
        for row in bad_cases[:5]:
            with st.expander(f"{row['bad_case_type']} | {row['question']}"):
                st.write(row)
    else:
        st.success("当前评测未发现 Bad Case。")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
    st.download_button(
        "下载评测 CSV",
        data=output.getvalue().encode("utf-8-sig"),
        file_name="rag_eval_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_recent_rag_logs() -> None:
    with st.expander("最近 RAG run 日志", expanded=False):
        try:
            logs = list_rag_logs(limit=20)
        except httpx.HTTPError as exc:
            st.caption(f"日志读取失败：{exc}")
            return
        if not logs:
            st.caption("暂无 RAG 日志。")
            return
        st.dataframe(logs, use_container_width=True, hide_index=True)


def _render_research_history() -> None:
    with st.expander("历史调研记录", expanded=False):
        try:
            runs = list_research_runs()
        except httpx.HTTPError as exc:
            st.caption(f"历史记录读取失败：{exc}")
            return

        if not runs:
            st.caption("暂无历史 Agent run。")
            return

        for run in runs:
            title = f"{run.get('created_at', '')} | {run.get('run_id', '')}"
            st.write(title)
            st.caption(
                f"evidence={run.get('evidence_count', 0)} | "
                f"warnings={run.get('warning_count', 0)} | {run.get('task', '')[:120]}"
            )
            if run.get("report_path"):
                st.caption(f"report: {run.get('report_path')}")


def main() -> None:
    init_page()
    top_k, allow_fallback, answer_style = render_sidebar()
    rag_tab, agent_tab, eval_tab = st.tabs(["RAG 问答", "论文调研 Agent", "RAG 评测"])
    with rag_tab:
        render_rag_chat(top_k=top_k, allow_fallback=allow_fallback, answer_style=answer_style)
    with agent_tab:
        render_agent_page(top_k=top_k)
    with eval_tab:
        render_rag_evaluation(top_k=top_k, answer_style=answer_style)


if __name__ == "__main__":
    main()
