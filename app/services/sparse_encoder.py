"""Deterministic lexical sparse vectors for Qdrant's native sparse index."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from qdrant_client.models import SparseVector


def tokenize_sparse(text: str) -> list[str]:
    """Tokenize English identifiers and Chinese n-grams without external models."""
    ascii_terms = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{1,}", text.lower())
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    terms: list[str] = []
    chinese_ngrams: list[str] = []
    for span in chinese_terms:
        # Keep the full span and add short overlapping n-grams so a query can
        # match a phrase embedded in a longer Chinese sentence.
        chinese_ngrams.append(span)
        for size in range(2, min(5, len(span)) + 1):
            chinese_ngrams.extend(span[index : index + size] for index in range(len(span) - size + 1))
    for term in [*ascii_terms, *chinese_ngrams]:
        if term not in terms:
            terms.append(term)
    return terms


def encode_sparse(text: str) -> SparseVector:
    """Build a deterministic TF sparse vector.

    The collection uses Qdrant's IDF modifier, so the stored TF vector is
    reweighted by collection statistics at query time. Hashing keeps the
    index self-contained and stable across ingestion processes.
    """
    terms = tokenize_sparse(text)
    counts = Counter(terms)
    indexed = sorted((_term_index(term), count) for term, count in counts.items())
    return SparseVector(
        indices=[index for index, _ in indexed],
        values=[round(1.0 + math.log(count), 6) for _, count in indexed],
    )


def _term_index(term: str) -> int:
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest()
    # Keep the index in Qdrant's non-negative uint32 range and avoid zero.
    return max(1, int.from_bytes(digest, "big") & 0x7FFFFFFF)
