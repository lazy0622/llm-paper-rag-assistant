from pathlib import Path
import re
import uuid

from fastapi import APIRouter, UploadFile
from fastapi import HTTPException

from app.config import settings
from app.schemas import DocumentActionResponse, DocumentSummary, IngestResponse, IngestionJobResponse
from app.services.ingestion_jobs import enqueue_document_ingestion, get_ingestion_job, list_ingestion_jobs
from app.services.vector_store import delete_document, list_indexed_documents

router = APIRouter()


@router.get("/jobs", response_model=list[IngestionJobResponse])
def ingestion_jobs(limit: int = 50) -> list[IngestionJobResponse]:
    return [IngestionJobResponse(**job) for job in list_ingestion_jobs(limit=limit)]


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse)
def ingestion_job(job_id: str) -> IngestionJobResponse:
    job = get_ingestion_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return IngestionJobResponse(**job)


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

    job = enqueue_document_ingestion(
        file_name=safe_name,
        source_path=target_path,
        target_path=target_path,
        replace_on_success=False,
    )
    return IngestResponse(
        file_name=safe_name,
        chunks=job["chunks"],
        status=job["status"],
        job_id=job["job_id"],
        document_id=job.get("document_id"),
        content_hash=job.get("content_hash"),
        error=job.get("error"),
    )


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile) -> IngestResponse:
    file_name = _safe_file_name(file.filename or "uploaded.pdf")
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".docx", ".md", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, Markdown, and TXT files are supported.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = settings.upload_dir / file_name
    temp_path = settings.upload_dir / f".{file_name}.{uuid.uuid4().hex}.tmp{suffix}"
    temp_path.write_bytes(await file.read())

    try:
        # Parsing and embedding happen in a durable background job. The
        # previous indexed version remains available until the job succeeds.
        job = enqueue_document_ingestion(
            file_name=file_name,
            source_path=temp_path,
            target_path=target_path,
            replace_on_success=True,
        )
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion job creation failed: {exc}") from exc

    return IngestResponse(
        file_name=file_name,
        chunks=job["chunks"],
        status=job["status"],
        job_id=job["job_id"],
        document_id=job.get("document_id"),
        content_hash=job.get("content_hash"),
        error=job.get("error"),
    )


def _safe_file_name(file_name: str) -> str:
    # 避免上传文件名里带路径或特殊字符，防止覆盖非预期文件。
    name = Path(file_name).name
    return re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]", "_", name)
