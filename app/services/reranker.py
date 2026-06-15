import re

from app.config import settings


def rerank_candidates(semantic_chunks: list[dict], keyword_chunks: list[dict]) -> list[dict]:
    """Merge and rerank candidates with the configured reranker provider.

    v1 only ships a rule-based reranker. The function boundary is intentionally
    stable so a future cross-encoder reranker can replace the internals without
    changing vector_store or API code.
    """
    provider = settings.reranker_provider.lower().strip()
    if provider != "rule":
        # Keep the system runnable even when someone experiments with config.
        provider = "rule"
    return _rule_rerank(semantic_chunks, keyword_chunks)


def keyword_overlap_score(query: str, content: str) -> float:
    query_terms = tokenize_for_keyword_score(query)
    if not query_terms:
        return 0.0
    normalized_content = content.lower()
    hits = sum(1 for term in query_terms if term in normalized_content)
    return hits / len(query_terms)


def tokenize_for_keyword_score(text: str) -> list[str]:
    ascii_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", text.lower())
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    terms = []
    for term in [*ascii_terms, *chinese_terms]:
        if term not in terms:
            terms.append(term)
    return terms


def _rule_rerank(semantic_chunks: list[dict], keyword_chunks: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for chunk in semantic_chunks:
        key = chunk.get("chunk_id") or _fallback_chunk_key(chunk)
        merged[key] = chunk

    for chunk in keyword_chunks:
        key = chunk.get("chunk_id") or _fallback_chunk_key(chunk)
        if key not in merged:
            merged[key] = chunk
            continue

        current = merged[key]
        current["keyword_score"] = max(current.get("keyword_score") or 0, chunk.get("keyword_score") or 0)
        current["keyword_rank"] = chunk.get("keyword_rank")
        current["retrieval_source"] = "hybrid"
        current["rerank_score"] = _rule_rerank_score(current.get("score") or 0, current.get("keyword_score") or 0)

    for chunk in merged.values():
        chunk["rerank_score"] = _rule_rerank_score(chunk.get("score") or 0, chunk.get("keyword_score") or 0)
        chunk["rrf_score"] = _rrf_score(chunk.get("semantic_rank"), chunk.get("keyword_rank"))
        chunk["final_score"] = round((chunk.get("rerank_score") or 0) * 0.7 + chunk["rrf_score"] * 0.3, 6)

    return sorted(
        merged.values(),
        key=lambda chunk: (
            chunk.get("final_score", chunk.get("rerank_score", chunk.get("score", 0))),
            chunk.get("rerank_score", 0),
        ),
        reverse=True,
    )


def _rule_rerank_score(semantic_score: float, keyword_score: float) -> float:
    if not settings.enable_keyword_rerank:
        return semantic_score
    return round(semantic_score * 0.75 + keyword_score * 0.25, 6)


def _rrf_score(semantic_rank: int | None, keyword_rank: int | None, k: int = 60) -> float:
    score = 0.0
    if semantic_rank:
        score += 1 / (k + semantic_rank)
    if keyword_rank:
        score += 1 / (k + keyword_rank)
    return round(score, 6)


def _fallback_chunk_key(chunk: dict) -> str:
    return f"{chunk.get('file_name')}:{chunk.get('page')}:{chunk.get('content', '')[:80]}"
