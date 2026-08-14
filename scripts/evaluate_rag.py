import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evaluation import answer_overlap_metrics, parse_expected_list, parse_expected_pages, retrieval_metrics


def evaluate(
    limit: int | None = None,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    run_label: str | None = None,
) -> Path:
    """Run the QA set and save retrieval/answer evidence for interview review."""
    # Keep metric helpers importable in lightweight CI environments. The
    # runtime-only integrations are loaded only when a real evaluation starts.
    from app.services.query_rewriter import rewrite_query_for_retrieval
    from app.services.evidence import validate_citation_markers
    from app.services.rag_chain import answer_with_context, answer_with_fallback, is_strict_knowledge_question, needs_fallback
    from app.services.vector_store import search_chunks

    input_path = input_path or PROJECT_ROOT / "data" / "eval" / "qa_pairs.csv"
    output_dir = output_dir or PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rag_eval_results.csv"
    summary_path = output_dir / "project_metrics.md"
    summary_json_path = output_dir / "rag_eval_summary.json"
    manifest_path = output_dir / "rag_eval_manifest.json"

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if limit is not None:
        rows = rows[:limit]

    results = []
    for row in rows:
        started = time.perf_counter()
        run_id = f"eval-{uuid.uuid4().hex}"
        question = row["question"]
        top_k = int(row.get("top_k") or 5)
        rewritten_query = rewrite_query_for_retrieval(question, run_id=run_id)
        chunks = search_chunks(rewritten_query, top_k=top_k, run_id=run_id)
        allow_fallback = not is_strict_knowledge_question(question)

        if not chunks:
            answer = (
                answer_with_fallback(question, "standard", run_id=run_id)
                if allow_fallback
                else "资料中没有明确依据。当前问题要求严格基于知识库回答。"
            )
            answer_mode = "fallback" if allow_fallback else "no_answer"
        else:
            answer = answer_with_context(question, chunks, "standard", run_id=run_id)
            if needs_fallback(answer):
                answer = (
                    answer_with_fallback(question, "standard", run_id=run_id)
                    if allow_fallback
                    else "资料中没有明确依据。当前问题要求严格基于知识库回答。"
                )
                answer_mode = "fallback" if allow_fallback else "no_answer"
            else:
                answer_mode = "grounded"

        source_chunks = chunks if answer_mode == "grounded" else []
        citation_warnings = validate_citation_markers(answer, len(source_chunks)) if answer_mode == "grounded" else []
        source_files = sorted({chunk.get("file_name", "") for chunk in source_chunks if chunk.get("file_name")})
        expected_source = row.get("source_file", "")
        expected_source_files = parse_expected_list(row.get("gold_source_files"))
        if expected_source and expected_source not in expected_source_files:
            expected_source_files.insert(0, expected_source)
        gold_chunk_ids = parse_expected_list(row.get("gold_chunk_ids"))
        gold_pages = parse_expected_pages(row.get("gold_pages"))
        # Retrieval metrics use ranked results even when the final answer
        # abstains. Cited sources remain restricted to grounded answers.
        top_source = chunks[0] if chunks else {}
        top_score = top_source.get("score")
        retrieval = retrieval_metrics(
            chunks,
            expected_source_files=expected_source_files,
            gold_chunk_ids=gold_chunk_ids,
            gold_pages=gold_pages,
            top_k=top_k,
        )
        citation = retrieval_metrics(
            source_chunks,
            expected_source_files=expected_source_files,
            gold_chunk_ids=gold_chunk_ids,
            gold_pages=gold_pages,
            top_k=top_k,
        )
        answer_metrics = answer_overlap_metrics(row.get("expected_answer", ""), answer)
        source_hit = retrieval["source_hit"]
        bad_case_type = _classify_bad_case(
            answer_mode=answer_mode,
            source_count=len(source_chunks),
            source_hit=source_hit,
            top_score=top_score,
            expected_source=expected_source,
            gold_chunk_hit=_gold_chunk_hit(retrieval, gold_chunk_ids),
            answer_token_f1=answer_metrics["answer_token_f1"],
            has_expected_answer=bool(row.get("expected_answer", "").strip()),
            citation_warnings=citation_warnings,
        )

        results.append(
            {
                "question": question,
                "rewritten_query": rewritten_query,
                "expected_answer": row.get("expected_answer", ""),
                "source_file": expected_source,
                "gold_source_files": ";".join(expected_source_files),
                "gold_chunk_ids": ";".join(gold_chunk_ids),
                "gold_pages": ";".join(str(page) for page in gold_pages),
                "difficulty": row.get("difficulty", ""),
                "top_k": top_k,
                "answer": answer,
                "answer_mode": answer_mode,
                "source_count": len(source_chunks),
                "source_files": ";".join(source_files),
                "source_hit": source_hit,
                "retrieval_source_hit": source_hit,
                "retrieved_relevant_count": retrieval["retrieved_relevant_count"],
                "retrieval_precision_at_k": retrieval["retrieval_precision_at_k"],
                "retrieval_recall_at_k": retrieval["retrieval_recall_at_k"],
                "retrieval_mrr": retrieval["retrieval_mrr"],
                "first_relevant_rank": retrieval["first_relevant_rank"],
                "ground_truth_level": retrieval["ground_truth_level"],
                "citation_precision_at_k": citation["retrieval_precision_at_k"],
                "citation_recall_at_k": citation["retrieval_recall_at_k"],
                "citation_warnings": ";".join(citation_warnings),
                "citation_marker_valid": not citation_warnings if answer_mode == "grounded" else None,
                **answer_metrics,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "top_score": _format_score(top_score),
                "top_rerank_score": _format_score(top_source.get("rerank_score")),
                "top_keyword_score": _format_score(top_source.get("keyword_score")),
                "top_final_score": _format_score(top_source.get("final_score")),
                "top_retrieval_source": top_source.get("retrieval_source", ""),
                "top_reranker_provider": top_source.get("reranker_provider", ""),
                "bad_case_type": bad_case_type,
            }
        )

    fieldnames = [
        "question",
        "rewritten_query",
        "expected_answer",
        "source_file",
        "gold_source_files",
        "gold_chunk_ids",
        "gold_pages",
        "difficulty",
        "top_k",
        "answer",
        "answer_mode",
        "source_count",
        "source_files",
        "source_hit",
        "retrieval_source_hit",
        "retrieved_relevant_count",
        "retrieval_precision_at_k",
        "retrieval_recall_at_k",
        "retrieval_mrr",
        "first_relevant_rank",
        "ground_truth_level",
        "citation_precision_at_k",
        "citation_recall_at_k",
        "citation_warnings",
        "citation_marker_valid",
        "answer_token_precision",
        "answer_token_recall",
        "answer_token_f1",
        "latency_ms",
        "top_score",
        "top_rerank_score",
        "top_keyword_score",
        "top_final_score",
        "top_retrieval_source",
        "top_reranker_provider",
        "bad_case_type",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    evaluated_top_k = max((int(row.get("top_k") or 5) for row in results), default=5)
    summary = _build_summary(results, input_path=input_path, top_k=evaluated_top_k)
    summary["run"] = _build_run_metadata(input_path=input_path, run_label=run_label)
    _write_summary(results, summary_path, summary)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(summary["run"], ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _format_score(score: float | None) -> str:
    return "" if score is None else f"{score:.4f}"


def _classify_bad_case(
    *,
    answer_mode: str,
    source_count: int,
    source_hit: bool,
    top_score: float | None,
    expected_source: str,
    gold_chunk_hit: bool | None = None,
    answer_token_f1: float | None = None,
    has_expected_answer: bool = False,
    citation_warnings: list[str] | None = None,
) -> str:
    if answer_mode == "fallback":
        return "fallback_triggered"
    if answer_mode == "no_answer":
        return "strict_no_answer"
    if source_count == 0:
        return "no_sources"
    if citation_warnings:
        return "citation_marker_invalid"
    if expected_source and not source_hit:
        return "expected_source_missed"
    if gold_chunk_hit is False:
        return "gold_chunk_missed"
    if top_score is not None and top_score < 0.45:
        return "low_top_score"
    if has_expected_answer and answer_token_f1 is not None and answer_token_f1 < 0.1:
        return "low_answer_overlap"
    return "ok"


def _gold_chunk_hit(metrics: dict, gold_chunk_ids: list[str]) -> bool | None:
    if not gold_chunk_ids:
        return None
    return metrics["retrieved_relevant_count"] > 0


def _build_summary(results: list[dict], *, input_path: Path, top_k: int) -> dict:
    total = len(results)
    grounded = sum(1 for row in results if row["answer_mode"] == "grounded")
    fallback = sum(1 for row in results if row["answer_mode"] == "fallback")
    no_answer = sum(1 for row in results if row["answer_mode"] == "no_answer")
    source_hit = sum(1 for row in results if row["source_hit"] is True)
    bad_cases = [row for row in results if row["bad_case_type"] != "ok"]
    retrieval_sources = _count_by_key(results, "top_retrieval_source")
    reranker_providers = _count_by_key(results, "top_reranker_provider")
    bad_case_counts = _count_by_key(results, "bad_case_type")
    citation_warning_count = sum(1 for row in results if row.get("citation_warnings"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "top_k": top_k,
        "total": total,
        "grounded": grounded,
        "fallback": fallback,
        "no_answer": no_answer,
        "source_hit_count": source_hit,
        "source_hit_rate": _average([1.0 if row["source_hit"] else 0.0 for row in results]),
        "retrieval_precision_at_k": _average_metric(results, "retrieval_precision_at_k"),
        "retrieval_recall_at_k": _average_metric(results, "retrieval_recall_at_k"),
        "retrieval_mrr": _average_metric(results, "retrieval_mrr"),
        "citation_precision_at_k": _average_metric(results, "citation_precision_at_k"),
        "citation_recall_at_k": _average_metric(results, "citation_recall_at_k"),
        "answer_token_f1": _average_metric(results, "answer_token_f1"),
        "latency_ms_avg": _average_metric(results, "latency_ms"),
        "latency_ms_p95": _percentile([row["latency_ms"] for row in results], 0.95),
        "bad_case_count": len(bad_cases),
        "citation_warning_count": citation_warning_count,
        "retrieval_sources": retrieval_sources,
        "reranker_providers": reranker_providers,
        "bad_case_counts": bad_case_counts,
    }


def _build_run_metadata(*, input_path: Path, run_label: str | None) -> dict:
    """Capture the configuration needed to reproduce an evaluation run."""
    from app.config import settings

    return {
        "run_label": run_label or "default",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_revision": _git_revision(),
        "reranker_provider": settings.reranker_provider,
        "reranker_model": settings.reranker_model,
        "reranker_batch_size": settings.reranker_batch_size,
        "embedding_model": settings.ollama_embedding_model,
        "chat_model": settings.ollama_chat_model,
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_hybrid_collection": settings.qdrant_hybrid_collection,
        "dense_vector_name": settings.dense_vector_name,
        "sparse_vector_name": settings.sparse_vector_name,
        "enable_native_sparse_search": settings.enable_native_sparse_search,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "enable_contextual_chunk_embedding": settings.enable_contextual_chunk_embedding,
        "top_k": settings.top_k,
        "score_threshold": settings.score_threshold,
        "enable_query_rewrite": settings.enable_query_rewrite,
        "enable_hybrid_search": settings.enable_hybrid_search,
        "enable_keyword_rerank": settings.enable_keyword_rerank,
    }


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_summary(results: list[dict], summary_path: Path, summary: dict) -> None:
    total = summary["total"]
    grounded = summary["grounded"]
    fallback = summary["fallback"]
    no_answer = summary["no_answer"]
    source_hit = summary["source_hit_count"]
    bad_cases = [row for row in results if row["bad_case_type"] != "ok"]
    retrieval_sources = summary["retrieval_sources"]
    reranker_providers = summary["reranker_providers"]
    bad_case_counts = summary["bad_case_counts"]

    lines = [
        "# RAG 评测与 Bad Case 分析",
        "",
        "本报告由 `scripts/evaluate_rag.py` 生成，用于记录 RAG 的检索命中、Query Rewrite、回答模式、Hybrid/Rerank 分数和引用来源问题。",
        "",
        "## 可复现配置",
        "",
        f"- run_label：{summary['run']['run_label']}",
        f"- git_revision：{summary['run']['git_revision'] or 'unknown'}",
        f"- reranker_provider：{summary['run']['reranker_provider']}",
        f"- reranker_model：{summary['run']['reranker_model']}",
        f"- embedding_model：{summary['run']['embedding_model']}",
        f"- chat_model：{summary['run']['chat_model']}",
        f"- chunk_size / overlap：{summary['run']['chunk_size']} / {summary['run']['chunk_overlap']}",
        f"- query_rewrite / hybrid：{summary['run']['enable_query_rewrite']} / {summary['run']['enable_hybrid_search']}",
        "",
        "## 汇总指标",
        "",
        f"- 评测问题数：{total}",
        f"- grounded：{grounded}",
        f"- fallback：{fallback}",
        f"- no_answer：{no_answer}",
        f"- 命中期望来源：{source_hit}（{_format_percent(summary['source_hit_rate'])}）",
        f"- Retrieval Precision@K：{_format_metric(summary['retrieval_precision_at_k'])}",
        f"- Retrieval Recall@K：{_format_metric(summary['retrieval_recall_at_k'])}",
        f"- Retrieval MRR：{_format_metric(summary['retrieval_mrr'])}",
        f"- Citation Precision@K：{_format_metric(summary['citation_precision_at_k'])}",
        f"- Citation Recall@K：{_format_metric(summary['citation_recall_at_k'])}",
        f"- Answer Token F1（词面回归指标，不等于语义正确率）：{_format_metric(summary['answer_token_f1'])}",
        f"- 平均延迟：{_format_metric(summary['latency_ms_avg'])} ms",
        f"- P95 延迟：{_format_metric(summary['latency_ms_p95'])} ms",
        f"- 结构化引用告警：{summary['citation_warning_count']}",
        f"- Bad Case 数：{len(bad_cases)}",
        "",
        "## Top 来源类型",
        "",
    ]

    lines.extend(f"- {key or 'unknown'}：{value}" for key, value in retrieval_sources.items())
    lines.extend(["", "## Reranker 类型", ""])
    lines.extend(f"- {key or 'unknown'}：{value}" for key, value in reranker_providers.items())
    lines.extend(["", "## Bad Case 类型分布", ""])
    lines.extend(f"- {key}：{value}" for key, value in bad_case_counts.items())
    lines.extend(["", "## Bad Case 明细", ""])

    if not bad_cases:
        lines.append("- 暂无 Bad Case。")
    else:
        for index, row in enumerate(bad_cases, start=1):
            lines.extend(
                [
                    f"### {index}. {row['bad_case_type']}",
                    "",
                    f"- 问题：{row['question']}",
                    f"- 改写检索 query：{row['rewritten_query']}",
                    f"- 期望来源：{row['source_file']}",
                    f"- 实际来源：{row['source_files'] or 'none'}",
                    f"- answer_mode：{row['answer_mode']}",
                    f"- top_score：{row['top_score']}",
                    f"- top_rerank_score：{row['top_rerank_score']}",
                    f"- top_keyword_score：{row['top_keyword_score']}",
                    f"- top_final_score：{row['top_final_score']}",
                    f"- top_retrieval_source：{row['top_retrieval_source'] or 'unknown'}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 后续优化方向",
            "",
            "- 对 `expected_source_missed`：检查 chunk 切分、Embedding 模型、TopK、Query Rewrite、Hybrid Search 和 Rerank 策略。",
            "- 对 `low_top_score`：考虑优化 query rewrite、补充术语词表，或接入 cross-encoder reranker。",
            "- 对 `fallback_triggered`：检查知识库是否缺资料，或 Prompt 是否过早拒答。",
            "- 对 `no_sources`：检查文档是否正确入库，Qdrant collection 是否可用。",
            "- 对 `low_answer_overlap`：这是词面回归信号，不应单独作为答案正确结论，需补充语义评测或人工复核。",
            "- 优先给 QA 样本补充 `gold_chunk_ids` 和 `gold_pages`，否则当前指标只能做到文件级来源判断。",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _count_by_key(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or "unknown"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _average_metric(rows: list[dict], key: str) -> float | None:
    return _average([float(row[key]) for row in rows if row.get(key) not in (None, "")])


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return round(ordered[index], 2)


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the paper RAG pipeline with qa_pairs.csv.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument("--input", type=Path, default=None, help="Evaluation CSV path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for this run's artifacts.")
    parser.add_argument(
        "--reranker-provider",
        choices=("rule", "cross_encoder"),
        default=None,
        help="Override RERANKER_PROVIDER for this process; useful for A/B runs.",
    )
    parser.add_argument("--run-label", default=None, help="Human-readable label stored in the run manifest.")
    args = parser.parse_args()

    if args.reranker_provider:
        os.environ["RERANKER_PROVIDER"] = args.reranker_provider

    output_path = evaluate(
        limit=args.limit,
        input_path=args.input,
        output_dir=args.output_dir,
        run_label=args.run_label,
    )
    print(f"Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
