from pathlib import Path

import fitz
from docx import Document


def load_document(path: Path) -> list[dict]:
    """Load a document into page-aware text blocks.

    这里保留页码/文件名，是为了后续回答时能做“引用溯源”，面试时也容易讲清楚。
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in {".md", ".txt"}:
        return _load_text(path)
    raise ValueError(f"Unsupported document type: {suffix}")


def _load_pdf(path: Path) -> list[dict]:
    pages = []
    with fitz.open(path) as pdf:
        for index, page in enumerate(pdf, start=1):
            # PDF 按页解析而不是整篇拼接，是为了让最终 sources 能精确到页码。
            # 这是 RAG 项目从“能回答”升级到“可溯源、可检查”的关键设计。
            text = page.get_text("text").strip()
            if text:
                pages.append({"file_name": path.name, "page": index, "text": text})
    return pages


def _load_docx(path: Path) -> list[dict]:
    document = Document(path)
    # Word 文档没有天然页码，这里先按整篇文档处理；后续可按标题/段落增强结构化切分。
    text = "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())
    return [{"file_name": path.name, "page": None, "text": text}] if text else []


def _load_text(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{"file_name": path.name, "page": None, "text": text}] if text else []
