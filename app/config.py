from pathlib import Path
import os

from pydantic_settings import BaseSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 本地服务经常跑在 localhost，如果系统代理接管了 httpx 请求，
# Ollama/Qdrant 可能会被误转发到 SOCKS/HTTP 代理导致连接失败。
for proxy_key in ("NO_PROXY", "no_proxy"):
    current = os.environ.get(proxy_key, "")
    local_hosts = ["localhost", "127.0.0.1", "::1"]
    missing = [host for host in local_hosts if host not in current]
    if missing:
        os.environ[proxy_key] = ",".join([part for part in [current, *missing] if part])


class Settings(BaseSettings):
    app_name: str = "LLM Paper RAG Assistant"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:3b"
    ollama_embedding_model: str = "mxbai-embed-large"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "llm_papers"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    score_threshold: float = 0.3
    enable_query_rewrite: bool = True
    query_rewrite_model: str = ""
    enable_hybrid_search: bool = True
    enable_keyword_rerank: bool = True
    reranker_provider: str = "rule"
    rerank_candidate_multiplier: int = 3
    keyword_candidate_limit: int = 30
    use_inference_gateway: bool = False
    inference_gateway_base_url: str = "http://127.0.0.1:8010"
    inference_gateway_api_key: str = "dev-local-key"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    rag_log_path: Path = PROJECT_ROOT / "reports" / "rag_chat_runs.jsonl"

    class Config:
        env_file = ".env"


settings = Settings()
