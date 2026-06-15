import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.query_rewriter import rewrite_query_for_retrieval
from app.services.rag_chain import answer_with_context, answer_with_fallback, is_strict_knowledge_question, needs_fallback
from app.services.vector_store import search_chunks


def evaluate(limit: int | None = None) -> Path:
    """Run the QA set and save retrieval/answer evidence for interview review."""
    input_path = PROJECT_ROOT / "data" / "eval" / "qa_pairs.csv"
    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rag_eval_results.csv"
    summary_path = output_dir / "project_metrics.md"

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    if limit is not None:
        rows = rows[:limit]

    results = []
    for row in rows:
        question = row["question"]
        rewritten_query = rewrite_query_for_retrieval(question)
        chunks = search_chunks(rewritten_query, top_k=5)
        allow_fallback = not is_strict_knowledge_question(question)

        if not chunks:
            answer = answer_with_fallback(question, "standard") if allow_fallback else "资料中没有明确依据。当前问题要求严格基于知识库回答。"
            answer_mode = "fallback" if allow_fallback else "no_answer"
        else:
            answer = answer_with_context(question, chunks, "standard")
            if needs_fallback(answer):
                answer = answer_with_fallback(question, "standard") if allow_fallback else "资料中没有明确依据。当前问题要求严格基于知识库回答。"
                answer_mode = "fallback" if allow_fallback else "no_answer"
            else:
                answer_mode = "grounded"

        source_chunks = chunks if answer_mode == "grounded" else []
        source_files = sorted({chunk.get("file_name", "") for chunk in source_chunks if chunk.get("file_name")})
        expected_source = row.get("source_file", "")
        top_source = source_chunks[0] if source_chunks else {}
        top_score = top_source.get("score")
        source_hit = bool(expected_source and expected_source in source_files)
        bad_case_type = _classify_bad_case(
            answer_mode=answer_mode,
            source_count=len(source_chunks),
            source_hit=source_hit,
            top_score=top_score,
            expected_source=expected_source,
        )

        results.append(
            {
                "question": question,
                "rewritten_query": rewritten_query,
                "expected_answer": row.get("expected_answer", ""),
                "source_file": expected_source,
                "difficulty": row.get("difficulty", ""),
                "answer": answer,
                "answer_mode": answer_mode,
                "source_count": len(source_chunks),
                "source_files": ";".join(source_files),
                "source_hit": source_hit,
                "top_score": _format_score(top_score),
                "top_rerank_score": _format_score(top_source.get("rerank_score")),
                "top_keyword_score": _format_score(top_source.get("keyword_score")),
                "top_final_score": _format_score(top_source.get("final_score")),
                "top_retrieval_source": top_source.get("retrieval_source", ""),
                "bad_case_type": bad_case_type,
            }
        )

    fieldnames = [
        "question",
        "rewritten_query",
        "expected_answer",
        "source_file",
        "difficulty",
        "answer",
        "answer_mode",
        "source_count",
        "source_files",
        "source_hit",
        "top_score",
        "top_rerank_score",
        "top_keyword_score",
        "top_final_score",
        "top_retrieval_source",
        "bad_case_type",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    _write_summary(results, summary_path)
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


def _write_summary(results: list[dict], summary_path: Path) -> None:
    total = len(results)
    grounded = sum(1 for row in results if row["answer_mode"] == "grounded")
    fallback = sum(1 for row in results if row["answer_mode"] == "fallback")
    no_answer = sum(1 for row in results if row["answer_mode"] == "no_answer")
    source_hit = sum(1 for row in results if row["source_hit"] is True)
    bad_cases = [row for row in results if row["bad_case_type"] != "ok"]
    retrieval_sources = _count_by_key(results, "top_retrieval_source")
    bad_case_counts = _count_by_key(results, "bad_case_type")

    lines = [
        "# RAG 评测与 Bad Case 分析",
        "",
        "本报告由 `scripts/evaluate_rag.py` 生成，用于记录 RAG 的检索命中、Query Rewrite、回答模式、Hybrid/Rerank 分数和引用来源问题。",
        "",
        "## 汇总指标",
        "",
        f"- 评测问题数：{total}",
        f"- grounded：{grounded}",
        f"- fallback：{fallback}",
        f"- no_answer：{no_answer}",
        f"- 命中期望来源：{source_hit}",
        f"- Bad Case 数：{len(bad_cases)}",
        "",
        "## Top 来源类型",
        "",
    ]

    lines.extend(f"- {key or 'unknown'}：{value}" for key, value in retrieval_sources.items())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the paper RAG pipeline with qa_pairs.csv.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    args = parser.parse_args()

    output_path = evaluate(limit=args.limit)
    print(f"Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
