import re
import threading
from math import exp
from typing import Any

from app.config import settings


_cross_encoder = None
_cross_encoder_name: str | None = None
_cross_encoder_lock = threading.Lock()


def rerank_candidates(
    semantic_chunks: list[dict],
    keyword_chunks: list[dict],
    query: str | None = None,
) -> list[dict]:
    """Merge and rerank candidates with the configured reranker provider.

    The default rule provider has no extra model dependency. A real
    cross-encoder is loaded lazily only when ``RERANKER_PROVIDER`` is set to
    ``cross_encoder``.
    """
    provider = settings.reranker_provider.lower().strip()
    if provider in {"cross_encoder", "cross-encoder", "crossencoder"}:
        return _cross_encoder_rerank(semantic_chunks, keyword_chunks, query=query)
    if provider not in {"", "rule"}:
        raise ValueError(
            f"Unsupported reranker_provider={settings.reranker_provider!r}. "
            "Expected 'rule' or 'cross_encoder'."
        )
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
    merged = _merge_candidates(semantic_chunks, keyword_chunks)
    for chunk in merged.values():
        chunk["reranker_provider"] = "rule"
        chunk["rerank_score"] = _rule_rerank_score(chunk.get("score") or 0, chunk.get("keyword_score") or 0)
        chunk["rrf_score"] = _rrf_score(chunk.get("semantic_rank"), chunk.get("keyword_rank"))
        chunk["final_score"] = round((chunk.get("rerank_score") or 0) * 0.7 + chunk["rrf_score"] * 0.3, 6)

    return _sort_candidates(merged.values())


def _cross_encoder_rerank(
    semantic_chunks: list[dict],
    keyword_chunks: list[dict],
    *,
    query: str | None,
) -> list[dict]:
    candidates = _merge_candidates(semantic_chunks, keyword_chunks)
    if not candidates:
        return []
    if not query:
        raise ValueError("A query is required for cross_encoder reranking.")

    model = _get_cross_encoder()
    pairs = [(query, chunk.get("content", "")) for chunk in candidates.values()]
    raw_scores = model.predict(
        pairs,
        batch_size=max(settings.reranker_batch_size, 1),
        show_progress_bar=False,
    )
    scores = _coerce_scores(raw_scores, expected_count=len(pairs))

    for chunk, raw_score in zip(candidates.values(), scores, strict=True):
        chunk["reranker_provider"] = "cross_encoder"
        chunk["rerank_score"] = _normalize_cross_encoder_score(raw_score)
        chunk["rrf_score"] = _rrf_score(chunk.get("semantic_rank"), chunk.get("keyword_rank"))
        chunk["final_score"] = round((chunk["rerank_score"] * 0.8) + (chunk["rrf_score"] * 0.2), 6)

    return _sort_candidates(candidates.values())


def _merge_candidates(semantic_chunks: list[dict], keyword_chunks: list[dict]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for chunk in semantic_chunks:
        key = chunk.get("chunk_id") or _fallback_chunk_key(chunk)
        merged[key] = dict(chunk)

    for chunk in keyword_chunks:
        key = chunk.get("chunk_id") or _fallback_chunk_key(chunk)
        if key not in merged:
            merged[key] = dict(chunk)
            continue

        current = merged[key]
        current["keyword_score"] = max(current.get("keyword_score") or 0, chunk.get("keyword_score") or 0)
        current["keyword_rank"] = chunk.get("keyword_rank")
        current["retrieval_source"] = "hybrid"
    return merged


def _sort_candidates(chunks) -> list[dict]:
    return sorted(
        chunks,
        key=lambda chunk: (
            chunk.get("final_score", chunk.get("rerank_score", chunk.get("score", 0))),
            chunk.get("rerank_score", 0),
        ),
        reverse=True,
    )


def _get_cross_encoder():
    global _cross_encoder, _cross_encoder_name
    model_name = settings.reranker_model.strip()
    if not model_name:
        raise RuntimeError("RERANKER_MODEL must be set when RERANKER_PROVIDER=cross_encoder.")
    if _cross_encoder is not None and _cross_encoder_name == model_name:
        return _cross_encoder

    with _cross_encoder_lock:
        if _cross_encoder is not None and _cross_encoder_name == model_name:
            return _cross_encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-Encoder reranking requires the optional dependency "
                "sentence-transformers. Install requirements-reranker.txt."
            ) from exc
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_name = model_name
        return _cross_encoder


def _coerce_scores(raw_scores: Any, expected_count: int) -> list[float]:
    if hasattr(raw_scores, "tolist"):
        raw_scores = raw_scores.tolist()
    if not isinstance(raw_scores, (list, tuple)):
        raw_scores = [raw_scores]
    if len(raw_scores) != expected_count:
        raise RuntimeError(
            f"Cross-Encoder returned {len(raw_scores)} scores for {expected_count} candidates."
        )

    scores = []
    for value in raw_scores:
        if isinstance(value, (list, tuple)):
            if not value:
                raise RuntimeError("Cross-Encoder returned an empty score vector.")
            value = value[-1]
        scores.append(float(value))
    return scores


def _normalize_cross_encoder_score(score: float) -> float:
    """Convert logits/probabilities into a comparable [0, 1] score."""
    if 0.0 <= score <= 1.0:
        return round(score, 6)
    if score >= 0:
        normalized = 1 / (1 + exp(-min(score, 50)))
    else:
        normalized = exp(max(score, -50)) / (1 + exp(max(score, -50)))
    return round(normalized, 6)


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
