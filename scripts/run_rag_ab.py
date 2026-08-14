"""Run reproducible RAG reranker A/B evaluations in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    "retrieval_precision_at_k",
    "retrieval_recall_at_k",
    "retrieval_mrr",
    "citation_precision_at_k",
    "citation_recall_at_k",
    "answer_token_f1",
    "latency_ms_avg",
    "latency_ms_p95",
    "bad_case_count",
)


def run_ab(
    *,
    providers: list[str],
    limit: int | None = None,
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or PROJECT_ROOT / "reports" / "rag_ab"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    for provider in providers:
        provider_dir = output_dir / provider
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "evaluate_rag.py"),
            "--reranker-provider",
            provider,
            "--run-label",
            provider,
            "--output-dir",
            str(provider_dir),
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        if input_path is not None:
            command.extend(["--input", str(input_path)])

        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"A/B evaluation failed for provider={provider}.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        summary_path = provider_dir / "rag_eval_summary.json"
        summaries[provider] = json.loads(summary_path.read_text(encoding="utf-8"))

    comparison = {
        "generated_at": summaries[providers[0]]["run"]["generated_at"],
        "providers": providers,
        "runs": summaries,
        "metric_comparison": _compare_metrics(summaries, providers),
    }
    json_path = output_dir / "ab_comparison.json"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(output_dir / "ab_comparison.md", comparison)
    return json_path


def _compare_metrics(summaries: dict[str, dict], providers: list[str]) -> dict[str, dict]:
    baseline = summaries[providers[0]]
    comparison: dict[str, dict] = {}
    for provider in providers:
        if provider == providers[0]:
            continue
        candidate = summaries[provider]
        provider_metrics: dict[str, dict] = {}
        for metric in DEFAULT_METRICS:
            base_value = baseline.get(metric)
            candidate_value = candidate.get(metric)
            if not isinstance(base_value, (int, float)) or not isinstance(candidate_value, (int, float)):
                provider_metrics[metric] = {"baseline": base_value, "candidate": candidate_value, "delta": None}
                continue
            provider_metrics[metric] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "delta": round(candidate_value - base_value, 4),
            }
        comparison[provider] = provider_metrics
    return comparison


def _write_markdown(path: Path, comparison: dict) -> None:
    providers = comparison["providers"]
    lines = [
        "# RAG Reranker A/B 对比",
        "",
        f"基线：`{providers[0]}`；候选：{', '.join(f'`{item}`' for item in providers[1:])}。",
        "",
        "| 指标 | 基线 | 候选 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for provider, metrics in comparison["metric_comparison"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {metric} ({provider}) | {values['baseline']} | "
                f"{values['candidate']} | {values['delta']} |"
            )
    lines.extend(
        [
            "",
            "说明：正向指标（Recall、MRR、Citation、Answer F1）越高越好；延迟和 Bad Case 越低越好。",
            "模型效果只有在真实 Ollama / Cross-Encoder 运行成功后才可写入简历。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated RAG reranker A/B evaluations.")
    parser.add_argument("--providers", default="rule,cross_encoder", help="Comma-separated provider order.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument("--input", type=Path, default=None, help="Evaluation CSV path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="A/B artifact directory.")
    args = parser.parse_args()

    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    if len(providers) < 2:
        parser.error("--providers must contain at least two providers, e.g. rule,cross_encoder")
    print(run_ab(providers=providers, limit=args.limit, input_path=args.input, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
