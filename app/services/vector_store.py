import unicodedata
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointIdsList,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings
from app.services.llm_client import embed_texts
from app.services.reranker import keyword_overlap_score, rerank_candidates, tokenize_for_keyword_score
from app.services.sparse_encoder import encode_sparse


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


def _ensure_hybrid_collection(client: QdrantClient, vector_size: int) -> None:
    """Create a named dense + native sparse collection for new indexes."""
    if client.collection_exists(settings.qdrant_hybrid_collection):
        return
    client.create_collection(
        collection_name=settings.qdrant_hybrid_collection,
        vectors_config={
            settings.dense_vector_name: VectorParams(size=vector_size, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            settings.sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF),
        },
    )


def upsert_chunks(chunks: list[dict], run_id: str | None = None) -> int:
    """Embed chunks and store vectors plus citation metadata in Qdrant."""
    if not chunks:
        return 0

    vectors = embed_texts([chunk.get("embedding_text", chunk["content"]) for chunk in chunks], run_id=run_id)
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
                "document_id": chunk.get("document_id"),
                "content_hash": chunk.get("content_hash"),
                "document_version": chunk.get("document_version"),
                "parser_version": chunk.get("parser_version"),
                "index_version": chunk.get("index_version"),
                "page": chunk.get("page"),
                "content": chunk["content"],
                "document_title": chunk.get("document_title"),
                "section": chunk.get("section"),
                "parent_chunk_id": chunk.get("parent_chunk_id"),
                "keyword_terms": tokenize_for_keyword_score(chunk["content"]),
            },
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)

    if settings.enable_native_sparse_search:
        _ensure_hybrid_collection(client, len(vectors[0]))
        hybrid_points = [
            PointStruct(
                id=point.id,
                vector={
                    settings.dense_vector_name: point.vector,
                    settings.sparse_vector_name: encode_sparse(chunk["content"]),
                },
                payload=point.payload,
            )
            for point, chunk in zip(points, chunks, strict=True)
        ]
        client.upsert(collection_name=settings.qdrant_hybrid_collection, points=hybrid_points)
    return len(points)


def search_chunks(query: str, top_k: int, run_id: str | None = None, file_name: str | None = None) -> list[dict]:
    """Search chunks with semantic retrieval, optional keyword recall, and pluggable rerank."""
    query_vector = embed_texts([query], run_id=run_id)[0]
    client = _client()
    if not client.collection_exists(settings.qdrant_collection) and not client.collection_exists(
        settings.qdrant_hybrid_collection
    ):
        return []

    query_filter = _file_filter(file_name) if file_name else None
    semantic_limit = top_k
    if settings.enable_keyword_rerank:
        semantic_limit = max(top_k, top_k * max(settings.rerank_candidate_multiplier, 1))

    use_native_sparse = _native_sparse_ready(client)
    semantic_collection = settings.qdrant_hybrid_collection if use_native_sparse else settings.qdrant_collection
    semantic_hits = _semantic_search(
        client,
        query_vector,
        limit=semantic_limit,
        query_filter=query_filter,
        collection_name=semantic_collection,
        vector_name=settings.dense_vector_name if use_native_sparse else None,
    )
    semantic_chunks = _hits_to_chunks(semantic_hits, query=query, retrieval_source="semantic")

    keyword_chunks = []
    if settings.enable_hybrid_search:
        if use_native_sparse:
            sparse_hits = _sparse_search(
                client,
                encode_sparse(query),
                limit=settings.keyword_candidate_limit,
                query_filter=query_filter,
            )
            keyword_chunks = _sparse_hits_to_chunks(sparse_hits)
        else:
            keyword_chunks = _keyword_search(client, query=query, file_name=file_name, limit=settings.keyword_candidate_limit)

    return rerank_candidates(semantic_chunks, keyword_chunks, query=query)[:top_k]


def _semantic_search(
    client: QdrantClient,
    query_vector: list[float],
    limit: int,
    query_filter: Filter | None,
    collection_name: str,
    vector_name: str | None,
):
    """Call Qdrant semantic search while staying compatible with old/new qdrant-client versions."""
    if hasattr(client, "query_points"):
        kwargs = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": limit,
            "with_payload": True,
            "query_filter": query_filter,
        }
        if vector_name:
            kwargs["using"] = vector_name
        result = client.query_points(**kwargs)
        return result.points
    return client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter,
        **({"using": vector_name} if vector_name else {}),
    )


def _sparse_search(client: QdrantClient, query_vector, limit: int, query_filter: Filter | None):
    """Search the native sparse vector index; Qdrant applies collection IDF."""
    if hasattr(client, "query_points"):
        result = client.query_points(
            collection_name=settings.qdrant_hybrid_collection,
            query=query_vector,
            using=settings.sparse_vector_name,
            limit=limit,
            with_payload=True,
            query_filter=query_filter,
        )
        return result.points
    return client.search(
        collection_name=settings.qdrant_hybrid_collection,
        query_vector=query_vector,
        using=settings.sparse_vector_name,
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
                "document_title": payload.get("document_title"),
                "section": payload.get("section"),
                "parent_chunk_id": payload.get("parent_chunk_id"),
                "score": semantic_score,
                "keyword_score": keyword_score,
                "rerank_score": semantic_score,
                "semantic_rank": rank,
                "keyword_rank": None,
                "retrieval_source": retrieval_source,
            }
        )
    return chunks


def _sparse_hits_to_chunks(hits) -> list[dict]:
    chunks = []
    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        content = payload.get("content", "")
        if _is_low_quality_content(content):
            continue
        score = float(hit.score)
        chunks.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "file_name": payload.get("file_name", "unknown"),
                "page": payload.get("page"),
                "content": content,
                "document_title": payload.get("document_title"),
                "section": payload.get("section"),
                "parent_chunk_id": payload.get("parent_chunk_id"),
                "score": 0.0,
                "keyword_score": score,
                "rerank_score": score,
                "semantic_rank": None,
                "keyword_rank": rank,
                "retrieval_source": "sparse",
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
                    "document_title": payload.get("document_title"),
                    "section": payload.get("section"),
                    "parent_chunk_id": payload.get("parent_chunk_id"),
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
    collection_name = _active_document_collection(client)
    if not collection_name:
        return []

    documents: dict[str, dict] = {}
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            file_name = payload.get("file_name", "unknown")
            page = payload.get("page")
            summary = documents.setdefault(
                file_name,
                {
                    "file_name": file_name,
                    "chunks": 0,
                    "pages": set(),
                    "document_ids": set(),
                    "content_hashes": set(),
                },
            )
            summary["chunks"] += 1
            if isinstance(page, int):
                summary["pages"].add(page)
            if payload.get("document_id"):
                summary["document_ids"].add(payload["document_id"])
            if payload.get("content_hash"):
                summary["content_hashes"].add(payload["content_hash"])
        if next_page is None:
            break

    return [
        {
            "file_name": item["file_name"],
            "chunks": item["chunks"],
            "pages": sorted(item["pages"]),
            "document_id": sorted(item["document_ids"])[-1] if item["document_ids"] else None,
            "content_hash": sorted(item["content_hashes"])[-1] if item["content_hashes"] else None,
        }
        for item in sorted(documents.values(), key=lambda value: value["file_name"])
    ]


def delete_document(file_name: str) -> int:
    """Delete all indexed chunks for one file from Qdrant."""
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0

    affected = _count_points_by_file(client, file_name, collection_name=settings.qdrant_collection)
    if affected == 0:
        return 0
    _delete_file_from_collection(client, file_name, settings.qdrant_collection)
    if client.collection_exists(settings.qdrant_hybrid_collection):
        _delete_file_from_collection(client, file_name, settings.qdrant_hybrid_collection)
    return affected


def prune_stale_document_versions(file_name: str, document_id: str, index_version: str) -> int:
    """Delete old versions after the new document has been indexed successfully."""
    client = _client()
    removed = _prune_stale_in_collection(client, file_name, document_id, index_version, settings.qdrant_collection)
    if client.collection_exists(settings.qdrant_hybrid_collection):
        _prune_stale_in_collection(client, file_name, document_id, index_version, settings.qdrant_hybrid_collection)
    return removed


def _count_points_by_file(client: QdrantClient, file_name: str, *, collection_name: str) -> int:
    count = 0
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
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


def _delete_file_from_collection(client: QdrantClient, file_name: str, collection_name: str) -> None:
    client.delete(collection_name=collection_name, points_selector=_file_filter(file_name))


def _prune_stale_in_collection(
    client: QdrantClient,
    file_name: str,
    document_id: str,
    index_version: str,
    collection_name: str,
) -> int:
    if not client.collection_exists(collection_name):
        return 0
    stale_point_ids = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            scroll_filter=_file_filter(file_name),
            limit=100,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("document_id") != document_id or payload.get("index_version") != index_version:
                stale_point_ids.append(point.id)
        if next_page is None:
            break
    if not stale_point_ids:
        return 0
    client.delete(collection_name=collection_name, points_selector=PointIdsList(points=stale_point_ids))
    return len(stale_point_ids)


def migrate_legacy_index_to_hybrid(batch_size: int = 100) -> int:
    """Copy existing dense points into the named dense+sparse collection."""
    client = _client()
    if not client.collection_exists(settings.qdrant_collection):
        return 0
    next_page = None
    migrated = 0
    batch: list[PointStruct] = []
    while True:
        points, next_page = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=batch_size,
            offset=next_page,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            vector = point.vector
            if isinstance(vector, dict):
                vector = vector.get(settings.dense_vector_name) or vector.get("default")
            if not vector:
                continue
            _ensure_hybrid_collection(client, len(vector))
            payload = point.payload or {}
            batch.append(
                PointStruct(
                    id=point.id,
                    vector={
                        settings.dense_vector_name: vector,
                        settings.sparse_vector_name: encode_sparse(payload.get("content", "")),
                    },
                    payload=payload,
                )
            )
            if len(batch) >= batch_size:
                client.upsert(collection_name=settings.qdrant_hybrid_collection, points=batch)
                migrated += len(batch)
                batch.clear()
        if next_page is None:
            break
    if batch:
        client.upsert(collection_name=settings.qdrant_hybrid_collection, points=batch)
        migrated += len(batch)
    return migrated


def _active_document_collection(client: QdrantClient) -> str | None:
    if _native_sparse_ready(client):
        return settings.qdrant_hybrid_collection
    if client.collection_exists(settings.qdrant_collection):
        return settings.qdrant_collection
    return None


def _native_sparse_ready(client: QdrantClient) -> bool:
    """Use the sparse path only after the sidecar contains every legacy point."""
    if not settings.enable_native_sparse_search:
        return False
    if not client.collection_exists(settings.qdrant_collection):
        return False
    if not client.collection_exists(settings.qdrant_hybrid_collection):
        return False
    try:
        legacy_count = client.count(collection_name=settings.qdrant_collection, exact=True).count
        hybrid_count = client.count(collection_name=settings.qdrant_hybrid_collection, exact=True).count
    except Exception:
        return False
    return hybrid_count >= legacy_count and hybrid_count > 0


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
