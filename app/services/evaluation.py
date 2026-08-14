"""Deterministic metrics for the paper RAG evaluation harness.

The online RAG path still owns retrieval and answer generation.  This module
only evaluates their outputs, so metric logic can be tested without starting
Ollama or Qdrant.  Answer overlap is intentionally labelled as a lexical
proxy; it is not a replacement for a semantic judge or human review.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable


_TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")


def parse_expected_list(value: str | None) -> list[str]:
    """Parse semicolon/pipe separated CSV ground-truth values."""
    if not value:
        return []
    values = re.split(r"[;|]", value)
    return [item.strip() for item in values if item.strip()]


def parse_expected_pages(value: str | None) -> list[int]:
    """Parse page numbers from a CSV field while ignoring malformed values."""
    pages: list[int] = []
    for item in parse_expected_list(value):
        try:
            page = int(item)
        except (TypeError, ValueError):
            continue
        if page > 0 and page not in pages:
            pages.append(page)
    return pages


def tokenize_for_evaluation(text: str) -> list[str]:
    """Tokenize Chinese and English answers without a third-party tokenizer.

    Chinese runs use overlapping bigrams so paraphrases still receive a useful
    signal.  The resulting score is only a cheap regression indicator.
    """
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    tokens: list[str] = []
    for part in _TOKEN_PATTERN.findall(normalized):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def answer_overlap_metrics(expected_answer: str, actual_answer: str) -> dict[str, float | None]:
    """Return lexical precision/recall/F1 against the expected answer."""
    if not expected_answer.strip():
        return {
            "answer_token_precision": None,
            "answer_token_recall": None,
            "answer_token_f1": None,
        }

    expected_tokens = Counter(tokenize_for_evaluation(expected_answer))
    actual_tokens = Counter(tokenize_for_evaluation(actual_answer))
    expected_count = sum(expected_tokens.values())
    actual_count = sum(actual_tokens.values())
    overlap_count = sum((expected_tokens & actual_tokens).values())

    precision = overlap_count / actual_count if actual_count else 0.0
    recall = overlap_count / expected_count if expected_count else 0.0
    f1 = _f1(precision, recall)
    return {
        "answer_token_precision": _round_metric(precision),
        "answer_token_recall": _round_metric(recall),
        "answer_token_f1": _round_metric(f1),
    }


def retrieval_metrics(
    chunks: Iterable[dict[str, Any]],
    *,
    expected_source_files: Iterable[str] = (),
    gold_chunk_ids: Iterable[str] = (),
    gold_pages: Iterable[int] = (),
    top_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate ranked evidence against the strongest available ground truth.

    Ground-truth priority is exact chunk IDs, then file+page pairs, then file
    names.  Existing CSV cases only contain ``source_file`` and therefore keep
    their old source-level behaviour until finer annotations are added.
    """
    ranked = list(chunks)
    if top_k is not None:
        ranked = ranked[: max(top_k, 0)]

    source_files = {value for value in expected_source_files if value}
    chunk_ids = {value for value in gold_chunk_ids if value}
    pages = {value for value in gold_pages if isinstance(value, int) and value > 0}
    use_chunk_ground_truth = bool(chunk_ids)
    use_page_ground_truth = bool(pages and source_files)

    relevant_flags = [
        _is_relevant(
            chunk,
            source_files=source_files,
            chunk_ids=chunk_ids,
            pages=pages,
            use_chunk_ground_truth=use_chunk_ground_truth,
            use_page_ground_truth=use_page_ground_truth,
        )
        for chunk in ranked
    ]
    relevant_count = sum(relevant_flags)
    gold_count = len(chunk_ids) if use_chunk_ground_truth else len(pages) if use_page_ground_truth else len(source_files)
    first_relevant_rank = next(
        (index for index, is_relevant in enumerate(relevant_flags, start=1) if is_relevant),
        None,
    )

    unique_source_files = {str(chunk.get("file_name", "")) for chunk in ranked if chunk.get("file_name")}
    source_hit = bool(source_files & unique_source_files) if source_files else relevant_count > 0
    precision = relevant_count / len(ranked) if ranked else 0.0
    recall = relevant_count / gold_count if gold_count else None

    return {
        "source_hit": source_hit,
        "retrieved_relevant_count": relevant_count,
        "retrieval_precision_at_k": _round_metric(precision),
        "retrieval_recall_at_k": _round_metric(recall),
        "retrieval_mrr": _round_metric(1 / first_relevant_rank) if first_relevant_rank else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "ground_truth_level": "chunk" if use_chunk_ground_truth else "page" if use_page_ground_truth else "file",
    }


def _is_relevant(
    chunk: dict[str, Any],
    *,
    source_files: set[str],
    chunk_ids: set[str],
    pages: set[int],
    use_chunk_ground_truth: bool,
    use_page_ground_truth: bool,
) -> bool:
    if use_chunk_ground_truth:
        return str(chunk.get("chunk_id", "")) in chunk_ids
    if use_page_ground_truth:
        return chunk.get("file_name") in source_files and chunk.get("page") in pages
    return bool(chunk.get("file_name") in source_files)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 4)
