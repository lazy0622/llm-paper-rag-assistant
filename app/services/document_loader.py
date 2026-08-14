import hashlib
import re
from pathlib import Path

import fitz
from docx import Document


def load_document(path: Path, file_name: str | None = None) -> list[dict]:
    """Load a document into page-aware text blocks.

    这里保留页码/文件名，是为了后续回答时能做“引用溯源”，面试时也容易讲清楚。
    """
    metadata = _document_metadata(path, file_name=file_name)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, metadata)
    if suffix == ".docx":
        return _load_docx(path, metadata)
    if suffix in {".md", ".txt"}:
        return _load_text(path, metadata)
    raise ValueError(f"Unsupported document type: {suffix}")


def _document_metadata(path: Path, file_name: str | None = None) -> dict:
    """Build stable identity metadata from the uploaded bytes."""
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file_name": file_name or path.name,
        "document_id": f"doc-{content_hash[:24]}",
        "content_hash": content_hash,
        "document_version": content_hash[:12],
        "parser_version": "page-text-v2",
        "document_title": Path(file_name or path.name).stem,
        "document_type": path.suffix.lower().lstrip("."),
    }


def _load_pdf(path: Path, metadata: dict) -> list[dict]:
    pages = []
    with fitz.open(path) as pdf:
        pdf_title = (pdf.metadata or {}).get("title") or metadata["document_title"]
        for index, page in enumerate(pdf, start=1):
            # PDF 按页解析而不是整篇拼接，是为了让最终 sources 能精确到页码。
            # 这是 RAG 项目从“能回答”升级到“可溯源、可检查”的关键设计。
            text = page.get_text("text").strip()
            if text:
                pages.append(
                    {
                        **metadata,
                        "document_title": pdf_title.strip() or metadata["document_title"],
                        "section": _infer_section(text),
                        "page": index,
                        "text": text,
                    }
                )
    return pages


def _load_docx(path: Path, metadata: dict) -> list[dict]:
    document = Document(path)
    # Word 文档没有天然页码，这里先按整篇文档处理；后续可按标题/段落增强结构化切分。
    text = "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())
    return [{**metadata, "page": None, "section": _infer_section(text), "text": text}] if text else []


def _load_text(path: Path, metadata: dict) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [{**metadata, "page": None, "section": _infer_section(text), "text": text}] if text else []


def _infer_section(text: str) -> str | None:
    """Use a short heading-like first line as lightweight paper metadata."""
    for line in text.splitlines():
        candidate = re.sub(r"\s+", " ", line).strip(" #-\t")
        if not candidate:
            continue
        if len(candidate) <= 120 and not candidate.endswith(("。", ".", ";", "；", ":", "：")):
            return candidate
        break
    return None
