import re
from collections import Counter
from typing import Literal


EvidenceLevel = Literal[
    "strong_support",
    "partial_support",
    "background_support",
    "weak_support",
    "insufficient",
]


TOPIC_CLAIMS = {
    "RAG": [
        "RAG 的核心思想和要解决的问题是什么？",
        "RAG-Sequence 和 RAG-Token 等检索增强生成方式有什么差异？",
    ],
    "Self-RAG": [
        "Self-RAG 如何通过检索、生成和自我反思提升事实性？",
        "Self-RAG 的优势、局限和适用场景是什么？",
    ],
    "CRAG": [
        "CRAG 如何判断检索质量并纠正低质量检索结果？",
        "CRAG 相比普通 RAG 的工程启发是什么？",
    ],
    "GraphRAG": [
        "GraphRAG 如何利用图结构或社区摘要支持全局问题回答？",
        "GraphRAG 的优势、局限和适用场景是什么？",
    ],
    "ReAct": [
        "ReAct 如何结合 reasoning 和 acting 完成工具使用？",
        "ReAct 对 Agent 工具调用流程有什么启发？",
    ],
}


def split_claims(task: str, topics: list[str]) -> list[str]:
    """Split a broad research task into a few evidence-checkable claims.

    This mirrors the Nature citation workflow: do not search a whole paragraph as
    one blob. Split it into focused claims first, then retrieve and grade evidence.
    """
    claims: list[str] = []
    for topic in topics:
        claims.extend(TOPIC_CLAIMS.get(topic, [f"{topic} 的核心思想、优缺点和适用场景是什么？"]))

    if not claims:
        claims = [
            f"{task} 的核心问题是什么？",
            f"{task} 涉及哪些主要方法路线？",
            f"{task} 的局限和适用边界是什么？",
        ]

    return _deduplicate_strings(claims)[:6]


def annotate_sources(chunks: list[dict], query: str) -> list[dict]:
    """Attach evidence labels to source chunks without changing retrieval order."""
    return [{**chunk, **_grade_chunk(query, chunk)} for chunk in chunks]


def grade_claim_evidence(claim: str, chunks: list[dict]) -> dict:
    """Summarize whether retrieved chunks really support a claim."""
    annotated = annotate_sources(chunks, claim)
    if not annotated:
        return {
            "claim": claim,
            "evidence_level": "insufficient",
            "evidence_reason": "没有检索到可用片段，不能把该 claim 写成确定结论。",
            "source_files": [],
            "source_pages": [],
            "source_count": 0,
            "top_score": None,
        }

    level_rank = {
        "strong_support": 4,
        "partial_support": 3,
        "background_support": 2,
        "weak_support": 1,
        "insufficient": 0,
    }
    best = max(annotated, key=lambda item: level_rank.get(item.get("evidence_level", "weak_support"), 0))
    source_files = sorted({chunk.get("file_name", "") for chunk in annotated if chunk.get("file_name")})
    source_pages = sorted({chunk.get("page") for chunk in annotated if isinstance(chunk.get("page"), int)})
    top_score = max((float(chunk.get("score") or 0) for chunk in annotated), default=None)

    return {
        "claim": claim,
        "evidence_level": best["evidence_level"],
        "evidence_reason": best["evidence_reason"],
        "source_files": source_files,
        "source_pages": source_pages,
        "source_count": len(annotated),
        "top_score": top_score,
    }


def format_evidence_map(evidence_items: list[dict]) -> str:
    if not evidence_items:
        return "- 暂无证据分级记录。"

    lines = []
    for item in evidence_items:
        pages = ", ".join(str(page) for page in item.get("source_pages", [])) or "unknown"
        files = "; ".join(item.get("source_files", [])) or "none"
        score = item.get("top_score")
        score_text = "None" if score is None else f"{score:.4f}"
        lines.append(
            f"- Claim: {item['claim']}\n"
            f"  Evidence: {item['evidence_level']} | top_score={score_text} | "
            f"sources={item['source_count']} | files={files} | pages={pages}\n"
            f"  Reason: {item['evidence_reason']}"
        )
    return "\n".join(lines)


def check_overclaim(report: str, evidence_items: list[dict]) -> list[str]:
    """Flag claims that sound stronger than the available evidence."""
    warnings: list[str] = []
    weak_items = [
        item
        for item in evidence_items
        if item.get("evidence_level") in {"background_support", "weak_support", "insufficient"}
    ]
    strong_words = ["一定", "完全", "必然", "绝对", "显著提升", "全面优于", "最优", "生产级", "彻底解决"]
    used_words = [word for word in strong_words if word in report]

    if used_words and weak_items:
        warnings.append(
            "报告中出现较强表述 "
            + "、".join(used_words)
            + "，但部分 claim 只有背景/弱证据或资料不足，建议改成有边界的表达。"
        )

    for item in weak_items:
        warnings.append(
            f"Claim「{item['claim']}」当前证据等级为 {item['evidence_level']}，"
            "不建议写成确定性结论。"
        )

    if not any(item.get("evidence_level") == "strong_support" for item in evidence_items):
        warnings.append("本次调研没有 strong_support 证据，报告应定位为初步调研而不是定论。")

    return _deduplicate_strings(warnings)


def append_overclaim_section(report: str, warnings: list[str]) -> str:
    if not warnings:
        return report
    section = "\n\n## 过度声称风险与边界\n" + "\n".join(f"- {warning}" for warning in warnings)
    return report.rstrip() + section


def _grade_chunk(query: str, chunk: dict) -> dict:
    score = float(chunk.get("score") or 0)
    content = chunk.get("content", "")
    overlap = _keyword_overlap(query, content)

    if score >= 0.60 and overlap >= 0.12:
        level: EvidenceLevel = "strong_support"
        reason = f"检索分数较高（{score:.4f}），且与问题关键词覆盖较好。"
    elif score >= 0.48 or overlap >= 0.10:
        level = "partial_support"
        reason = f"片段与问题有直接关联，但仍需要结合上下文核验（score={score:.4f}）。"
    elif score >= 0.30:
        level = "background_support"
        reason = f"片段可作为背景线索，但不足以单独支撑强结论（score={score:.4f}）。"
    else:
        level = "weak_support"
        reason = f"相似度或关键词覆盖偏弱，只能作为弱相关线索（score={score:.4f}）。"

    return {
        "evidence_level": level,
        "evidence_reason": reason,
    }


def _keyword_overlap(query: str, content: str) -> float:
    query_terms = set(_tokens(query))
    if not query_terms:
        return 0.0
    content_terms = set(_tokens(content[:1200]))
    return len(query_terms & content_terms) / len(query_terms)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)]


def _deduplicate_strings(items: list[str]) -> list[str]:
    return list(Counter(item for item in items if item).keys())
