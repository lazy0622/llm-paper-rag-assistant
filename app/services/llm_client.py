import httpx

from app.config import settings


def embed_texts(texts: list[str], run_id: str | None = None) -> list[list[float]]:
    """Call Ollama embedding API and return vectors.

    默认直连 Ollama；启用统一推理网关后，RAG 项目不再关心底层模型服务来自 Ollama 还是其他后端。
    """
    if not texts:
        return []
    if settings.use_inference_gateway:
        return _embed_texts_via_gateway(texts, run_id=run_id)

    base_url = settings.ollama_base_url.rstrip("/")
    # trust_env=False 避免读取系统代理环境变量，保证本地 Ollama 请求直连 localhost。
    with httpx.Client(timeout=120, trust_env=False) as client:
        response = client.post(
            f"{base_url}/api/embed",
            json={"model": settings.ollama_embedding_model, "input": texts, "truncate": True},
        )
        if response.status_code == 404:
            # 新版 Ollama 推荐 /api/embed，旧版可能只有 /api/embeddings。
            return [_embed_one_legacy(client, base_url, text) for text in texts]
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])


def _embed_one_legacy(client: httpx.Client, base_url: str, text: str) -> list[float]:
    response = client.post(
        f"{base_url}/api/embeddings",
        json={"model": settings.ollama_embedding_model, "prompt": text},
    )
    response.raise_for_status()
    return response.json()["embedding"]


def chat_completion(prompt: str, run_id: str | None = None, model: str | None = None) -> str:
    """Call the local chat model through Ollama."""
    if settings.use_inference_gateway:
        return _chat_completion_via_gateway(prompt, run_id=run_id, model=model)

    base_url = settings.ollama_base_url.rstrip("/")
    # 这里只封装模型调用，RAG 的检索、拼 Prompt 和引用逻辑放在上层，便于替换模型服务。
    with httpx.Client(timeout=180, trust_env=False) as client:
        response = client.post(
            f"{base_url}/api/chat",
            json={
                "model": model or settings.ollama_chat_model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


def _gateway_headers(run_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {settings.inference_gateway_api_key}"}
    if run_id:
        # X-Request-ID lets the inference gateway and RAG layer share one trace id.
        headers["X-Request-ID"] = run_id
    return headers


def _embed_texts_via_gateway(texts: list[str], run_id: str | None = None) -> list[list[float]]:
    base_url = settings.inference_gateway_base_url.rstrip("/")
    with httpx.Client(timeout=120, trust_env=False) as client:
        response = client.post(
            f"{base_url}/v1/embeddings",
            headers=_gateway_headers(run_id),
            json={"model": settings.ollama_embedding_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in sorted(data.get("data", []), key=lambda item: item["index"])]


def _chat_completion_via_gateway(prompt: str, run_id: str | None = None, model: str | None = None) -> str:
    base_url = settings.inference_gateway_base_url.rstrip("/")
    with httpx.Client(timeout=180, trust_env=False) as client:
        response = client.post(
            f"{base_url}/v1/chat/completions",
            headers=_gateway_headers(run_id),
            json={
                "model": model or settings.ollama_chat_model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
