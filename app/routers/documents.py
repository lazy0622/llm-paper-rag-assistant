from pathlib import Path
import re

from fastapi import APIRouter, UploadFile
from fastapi import HTTPException

from app.config import settings
from app.schemas import DocumentActionResponse, DocumentSummary, IngestResponse
from app.services.document_loader import load_document
from app.services.splitter import split_pages
from app.services.vector_store import delete_document, list_indexed_documents, upsert_chunks

router = APIRouter()


@router.get("", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return [DocumentSummary(**document) for document in list_indexed_documents()]


@router.delete("/{file_name}", response_model=DocumentActionResponse)
def delete_indexed_document(file_name: str) -> DocumentActionResponse:
    affected = delete_document(_safe_file_name(file_name))
    if affected == 0:
        raise HTTPException(status_code=404, detail="Document not found in Qdrant index.")
    return DocumentActionResponse(file_name=file_name, affected_chunks=affected, status="deleted")


@router.post("/{file_name}/reindex", response_model=IngestResponse)
def reindex_document(file_name: str) -> IngestResponse:
    safe_name = _safe_file_name(file_name)
    target_path = settings.upload_dir / safe_name
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Original uploaded file not found.")

    try:
        delete_document(safe_name)
        pages = load_document(target_path)
        chunks = split_pages(pages)
        stored = upsert_chunks(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document reindex failed: {exc}") from exc

    if stored == 0:
        raise HTTPException(status_code=400, detail="No valid text chunks were extracted from the document.")
    return IngestResponse(file_name=safe_name, chunks=stored, status="reindexed")


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile) -> IngestResponse:
    file_name = _safe_file_name(file.filename or "uploaded.pdf")
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, Markdown, and TXT files are supported.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = settings.upload_dir / file_name
    target_path.write_bytes(await file.read())

    try:
        pages = load_document(target_path)
        chunks = split_pages(pages)
        stored = upsert_chunks(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc

    if stored == 0:
        raise HTTPException(status_code=400, detail="No valid text chunks were extracted from the document.")

    return IngestResponse(file_name=file_name, chunks=stored, status="indexed")


def _safe_file_name(file_name: str) -> str:
    # 避免上传文件名里带路径或特殊字符，防止覆盖非预期文件。
    name = Path(file_name).name
    return re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]", "_", name)
