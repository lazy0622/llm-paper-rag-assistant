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
        for index, text in enumerate(splitter.split_text(page["text"])):
            content = text.strip()
            if not content:
                continue
            raw_id = f"{page['file_name']}:{page.get('page')}:{index}:{content[:80]}"
            # chunk_id 基于内容和位置生成，重复上传同一文档时可以稳定覆盖同一批 point。
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "file_name": page["file_name"],
                    "page": page.get("page"),
                    "content": content,
                }
            )
    return chunks
