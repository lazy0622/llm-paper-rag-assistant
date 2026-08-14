import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def split_pages(pages: list[dict]) -> list[dict]:
    """Split parsed pages into retrievable chunks.

    chunk_size/overlap 会直接影响召回质量：太小会丢上下文，太大会引入噪声。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # 分隔符从大结构到小结构逐级回退，尽量优先保留段落/句子边界。
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )

    chunks: list[dict] = []
    for page in pages:
        index_version = _index_version(page)
        for index, text in enumerate(splitter.split_text(page["text"])):
            content = text.strip()
            if not content:
                continue
            document_id = page.get("document_id") or page["file_name"]
            raw_id = f"{document_id}:{page['file_name']}:{page.get('page')}:{index}:{content[:80]}"
            # chunk_id 基于内容和位置生成，重复上传同一文档时可以稳定覆盖同一批 point。
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": page["file_name"],
                    "document_id": page.get("document_id"),
                    "content_hash": page.get("content_hash"),
                    "document_version": page.get("document_version"),
                    "parser_version": page.get("parser_version"),
                    "index_version": index_version,
                    "page": page.get("page"),
                    "content": content,
                    "document_title": page.get("document_title") or page["file_name"],
                    "section": page.get("section"),
                    "parent_chunk_id": _parent_chunk_id(document_id, page.get("page"), page.get("section")),
                    "embedding_text": _embedding_text(page, content),
                }
            )
    return chunks


def _index_version(page: dict) -> str:
    """Identify the exact parser/chunker/embed configuration for an index."""
    raw = "|".join(
        [
            str(page.get("document_id") or page.get("file_name", "unknown")),
            str(page.get("parser_version") or "unknown"),
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            settings.ollama_embedding_model,
            str(settings.enable_contextual_chunk_embedding),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _parent_chunk_id(document_id: str, page: int | None, section: str | None) -> str:
    raw = f"{document_id}:{page}:{section or 'root'}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _embedding_text(page: dict, content: str) -> str:
    if not settings.enable_contextual_chunk_embedding:
        return content
    title = page.get("document_title") or page.get("file_name", "unknown")
    section = page.get("section") or "unknown section"
    return f"Document title: {title}\nSection: {section}\nContent:\n{content}"
