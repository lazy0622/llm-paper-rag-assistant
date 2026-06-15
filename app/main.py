from fastapi import FastAPI

from app.routers import agent, chat, documents, health

app = FastAPI(title="LLM Paper RAG Assistant")

app.include_router(health.router)
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(agent.router, prefix="/agent", tags=["agent"])
