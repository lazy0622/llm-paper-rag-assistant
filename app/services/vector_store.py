import unicodedata
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.config import settings
from app.services.llm_client import embed_texts
from app.services.reranker import keyword_overlap_score, rerank_candidates, tokenize_for_keyword_score


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    """Create the Qdrant collection when it does not exist."""
    if client.collection_exists(settings.qdrant_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert_chunks(chunks: list[dict], run_id: str | None = None) -> int:
    """Embed chunks and store vectors plus citation metadata in Qdrant."""
    if not chunks:
        return 0

    vectors = embed_texts([chunk["content"] for chunk in chunks], run_id=run_id)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    client = _client()
    _ensure_collection(client, len(vectors[0]))

    points = [
        PointStruct(
            # Stable ids let repeated ingestion update the same chunk instead of
            # creating duplicated vectors.
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
            vector=vector,
            payload={
                "chunk_id": chunk["chunk_id"],
                "file_name": chunk["file_name"],
                "page": chunk.get("page"),
                "content": chunk["content"],
                "keyword_terms": tokenize_for_keyword_score(chunk["content"]),
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def search_chunks(query: str, top_k: int, run_id: str | None = None, file_name: str | None = None) -> list[dict]:
    """Search chunks with semantic retrieval, optional keyword recall, and pluggable rerank."""
    query_vector = embed_texts([query], run_id=run_id)[0]
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        return []

    query_filter = _file_filter(file_name) if file_name else None
    semantic_limit = top_k
    if settings.enable_keyword_rerank:
        semantic_limit = max(top_k, top_k * max(settings.rerank_candidate_multiplier, 1))

    semantic_hits = _semantic_search(client, query_vector, limit=semantic_limit, query_filter=query_filter)
    semantic_chunks = _hits_to_chunks(semantic_hits, query=query, retrieval_source="semantic")

    keyword_chunks = []
    if settings.enable_hybrid_search:
        keyword_chunks = _keyword_search(client, query=query, file_name=file_name, limit=settings.keyword_candidate_limit)

    return rerank_candidates(semantic_chunks, keyword_chunks)[:top_k]


def _semantic_search(client: QdrantClient, query_vector: list[float], limit: int, query_filter: Filter | None):
    """Call Qdrant semantic search while staying compatible with old/new qdrant-client versions."""
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return result.points
    return client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
    )


def _hits_to_chunks(hits, query: str, retrieval_source: str) -> list[dict]:
    chunks = []
    for rank, hit in enumerate(hits, start=1):
        if float(hit.score) < settings.score_threshold:
            continue
        payload = hit.payload or {}
        content = payload.get("content", "")
        if _is_low_quality_content(content):
            continue

        semantic_score = float(hit.score)
        keyword_score = keyword_overlap_score(query, content)
        chunks.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "file_name": payload.get("file_name", "unknown"),
                "page": payload.get("page"),
                "content": content,
                "score": semantic_score,
                "keyword_score": keyword_score,
                "rerank_score": semantic_score,
                "semantic_rank": rank,
                "keyword_rank": None,
                "retrieval_source": retrieval_source,
            }
        )
    return chunks


def _keyword_search(client: QdrantClient, query: str, file_name: str | None, limit: int) -> list[dict]:
    """Lightweight keyword recall over Qdrant payloads."""
    if not tokenize_for_keyword_score(query):
        return []

    candidates = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=_file_filter(file_name) if file_name else None,
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            content = payload.get("content", "")
            if _is_low_quality_content(content):
                continue

            keyword_score = keyword_overlap_score(query, content)
            if keyword_score <= 0:
                continue
            candidates.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "file_name": payload.get("file_name", "unknown"),
                    "page": payload.get("page"),
                    "content": content,
                    "score": 0.0,
                    "keyword_score": keyword_score,
                    "rerank_score": keyword_score,
                    "semantic_rank": None,
                    "keyword_rank": None,
                    "retrieval_source": "keyword",
                }
            )
        if next_page is None:
            break

    candidates.sort(key=lambda chunk: chunk["keyword_score"], reverse=True)
    for rank, chunk in enumerate(candidates[:limit], start=1):
        chunk["keyword_rank"] = rank
    return candidates[:limit]


def list_indexed_documents() -> list[dict]:
    """Aggregate indexed file metadata from Qdrant payloads."""
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        return []

    documents: dict[str, dict] = {}
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            file_name = payload.get("file_name", "unknown")
            page = payload.get("page")
            summary = documents.setdefault(file_name, {"file_name": file_name, "chunks": 0, "pages": set()})
            summary["chunks"] += 1
            if isinstance(page, int):
                summary["pages"].add(page)
        if next_page is None:
            break

    return [
        {
            "file_name": item["file_name"],
            "chunks": item["chunks"],
            "pages": sorted(item["pages"]),
        }
        for item in sorted(documents.values(), key=lambda value: value["file_name"])
    ]


def delete_document(file_name: str) -> int:
    """Delete all indexed chunks for one file from Qdrant."""
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0

    affected = _count_points_by_file(client, file_name)
    if affected == 0:
        return 0
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=_file_filter(file_name),
    )
    return affected


def _count_points_by_file(client: QdrantClient, file_name: str) -> int:
    count = 0
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=_file_filter(file_name),
            limit=100,
            offset=next_page,
            with_payload=False,
            with_vectors=False,
        )
        count += len(points)
        if next_page is None:
            break
    return count


def _file_filter(file_name: str) -> Filter:
    return Filter(must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))])


def _is_low_quality_content(content: str) -> bool:
    """Filter PDF extraction noise before it enters retrieval results or prompts."""
    if not content.strip():
        return True

    sample = content[:500]
    control_count = sum(1 for char in sample if unicodedata.category(char).startswith("C") and char not in "\n\t\r")
    punctuation_allowlist = ".,;:!?()[]{}<>/-_+=*&%$#@'\"`~，。；：！？（）【】《》"
    weird_symbol_count = sum(
        1
        for char in sample
        if not char.isalnum() and not char.isspace() and char not in punctuation_allowlist
    )
    if control_count / max(len(sample), 1) > 0.01:
        return True
    if weird_symbol_count / max(len(sample), 1) > 0.18:
        return True
    return False
