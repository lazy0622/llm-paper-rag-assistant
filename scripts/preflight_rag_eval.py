"""Check local services and optional dependencies before a real RAG evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_service(name: str, url: str, path: str) -> dict:
    endpoint = f"{url.rstrip('/')}{path}"
    try:
        response = httpx.get(endpoint, timeout=5, trust_env=False)
        return {
            "name": name,
            "endpoint": endpoint,
            "ok": response.is_success,
            "status_code": response.status_code,
            "error": None if response.is_success else response.text[:200],
        }
    except httpx.HTTPError as exc:
        return {"name": name, "endpoint": endpoint, "ok": False, "status_code": None, "error": str(exc)}


def preflight(provider: str | None = None) -> dict:
    from app.config import settings

    selected_provider = provider or settings.reranker_provider
    checks = [
        check_service("qdrant", settings.qdrant_url, "/healthz"),
    ]
    if settings.use_inference_gateway:
        checks.append(check_service("inference_gateway", settings.inference_gateway_base_url, "/health"))
    else:
        checks.append(check_service("ollama", settings.ollama_base_url, "/api/tags"))

    if selected_provider == "cross_encoder":
        checks.append(
            {
                "name": "sentence_transformers",
                "endpoint": "python package",
                "ok": importlib.util.find_spec("sentence_transformers") is not None,
                "status_code": None,
                "error": None,
            }
        )

    return {
        "provider": selected_provider,
        "collection": settings.qdrant_collection,
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight RAG services before a real evaluation.")
    parser.add_argument("--provider", choices=("rule", "cross_encoder"), default=None)
    args = parser.parse_args()
    result = preflight(args.provider)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
