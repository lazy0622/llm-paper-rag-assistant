from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers import agent, chat, documents, health
from app.services.ingestion_jobs import recover_pending_ingestion_jobs, shutdown_ingestion_executor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    recover_pending_ingestion_jobs()
    yield
    shutdown_ingestion_executor()


app = FastAPI(title="LLM Paper RAG Assistant", lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])
