"""
Gateway launcher example using `simple-ai-gateway`.

USAGE:
  Copy this file to gateway_launcher.py (which is gitignored) and customise it
  for your own gateway. This lets you swap in a different gateway library, add
  auth middleware, or change routing logic without git tracking your changes.

  cp gateway_launcher.example.py gateway_launcher.py
"""

# NOTE: No `from __future__ import annotations` here — FastAPI resolves route parameter
# types by name from the module's global namespace. Postponed evaluation (PEP 563) would
# turn annotations into strings, causing FastAPI to fail to resolve locally-defined models.

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

from fastapi import Header, Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Full OpenAI-compatible request schema.
# The library's built-in ChatRequest only has messages/model/stream — fields
# like max_tokens are silently dropped when Pydantic parses the request body.
# We replace the /v1/chat/completions route with one that uses this model.
# ---------------------------------------------------------------------------

class _ChatMessage(BaseModel):
    role: str
    content: str


class FullChatRequest(BaseModel):
    messages: List[_ChatMessage]
    model: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list] = None
    n: Optional[int] = None


# ---------------------------------------------------------------------------
# Prometheus middleware (simple-ai-gateway has no /metrics endpoint).
# ---------------------------------------------------------------------------

def _attach_prometheus(app) -> None:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response

    request_duration = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency",
        ["handler"],
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10,
            30, 60, 120, 180, 300, 600, 900,
        ),
    )
    request_count = Counter(
        "http_requests_total",
        "HTTP requests",
        ["handler", "status"],
    )

    class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            path = request.url.path
            code = response.status_code
            status = f"{code // 100}xx" if 100 <= code < 600 else "other"
            request_duration.labels(handler=path).observe(elapsed)
            request_count.labels(handler=path, status=status).inc()
            return response

    app.add_middleware(PrometheusMetricsMiddleware)

    @app.get("/metrics")
    def prometheus_metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = Path(__file__).resolve().parent
    cfg = root / "gateway_config.yaml"
    os.environ.setdefault("SIMPLE_AI_GATEWAY_CONFIG", str(cfg))

    import httpx
    from simple_ai_gateway.backends import remote_backend
    from simple_ai_gateway.backends.remote_backend import RemoteBackend

    remote_backend.httpx = httpx

    # Upstream hardcodes timeout=20s; Modal cold start often exceeds that.
    _upstream_timeout = float(os.environ.get("SIMPLE_AI_GATEWAY_UPSTREAM_TIMEOUT", "600"))
    # If set, replaces JSON "model" sent upstream (lets gateway model names differ from
    # the served model name, e.g. gateway uses "modal_vllm" while SGLang serves "tinyllama").
    _upstream_model = os.environ.get("SIMPLE_AI_GATEWAY_UPSTREAM_MODEL", "").strip()

    async def _generate_with_long_timeout(self, chat_req):
        async with httpx.AsyncClient(follow_redirects=False, trust_env=False) as client:
            try:
                # exclude_none preserves explicit user values (e.g. max_tokens=20).
                payload = chat_req.model_dump(mode="json", exclude_none=True)
                if _upstream_model:
                    payload["model"] = _upstream_model
                timeout = httpx.Timeout(_upstream_timeout, connect=60.0)
                headers_post = {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                # Inject bearer token when the backend has an api_key set.
                # (e.g. Anyscale services require Authorization: Bearer <token>).
                # api_key is stamped onto RemoteBackend instances by the patched
                # get_backend_instance below — the library's factory ignores it.
                api_key = getattr(self, "_api_key", None)
                if api_key:
                    headers_post["Authorization"] = f"Bearer {api_key}"
                headers_get = {"Accept": "application/json"}
                resp = await client.post(
                    self.url, json=payload, headers=headers_post, timeout=timeout
                )
                # Two distinct Modal 303 patterns:
                #
                # A) web_endpoint result polling: 303 → Location has __modal_function_call_id
                #    → GET the result URL to poll for the completed response.
                #
                # B) web_server cold-start: 303 → same path, no call ID
                #    → re-POST after a short pause (server not ready yet).
                for _ in range(24):
                    if resp.status_code not in (301, 302, 303, 307, 308):
                        break
                    loc = (resp.headers.get("location") or "").strip()
                    if not loc:
                        break
                    next_url = urljoin(str(resp.request.url), loc)
                    client.cookies.extract_cookies(resp)
                    if resp.status_code == 303 and "__modal_function_call_id" in next_url:
                        resp = await client.get(next_url, headers=headers_get, timeout=timeout)
                    elif resp.status_code == 303:
                        await asyncio.sleep(3)
                        resp = await client.post(
                            next_url, json=payload, headers=headers_post, timeout=timeout
                        )
                    else:
                        resp = await client.post(
                            next_url, json=payload, headers=headers_post, timeout=timeout
                        )
                resp.raise_for_status()
                backend_data = resp.json()
                return backend_data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                snippet = (e.response.text or "").strip()[:1200]
                detail = str(e).strip() or repr(e)
                suffix = f" — {snippet}" if snippet else ""
                return f"Backend Error ({self.url}): {detail}{suffix}"
            except Exception as e:
                return f"Backend Error ({self.url}): {str(e).strip() or repr(e)}"

    RemoteBackend.generate = _generate_with_long_timeout

    from simple_ai_gateway.main import app as sgw_app

    _attach_prometheus(sgw_app)

    # Patch get_backend_instance to forward `api_key` from the YAML config
    # onto the backend instance, so _generate_with_long_timeout can use it.
    from simple_ai_gateway import backends as _backends_mod
    _orig_get_backend = _backends_mod.get_backend_instance

    def _get_backend_with_api_key(model_name, config):
        backend = _orig_get_backend(model_name, config)
        cfg = config["backends"].get(model_name)
        if not cfg:
            cfg = config["backends"][config["default_backend"]]
        api_key = cfg.get("api_key", "")
        if api_key:
            backend._api_key = api_key
        return backend

    _backends_mod.get_backend_instance = _get_backend_with_api_key

    # Replace the stock /v1/chat/completions route with FullChatRequest so that
    # max_tokens and other OpenAI fields aren't silently dropped at parse time.
    # Use _backends_mod.get_backend_instance (not a local import) so the patched
    # version above (which forwards api_key) is always called.
    from simple_ai_gateway.main import (
        CONFIG,
        generate_stream,
        limiter,
    )

    sgw_app.router.routes = [
        r for r in sgw_app.router.routes
        if not (isinstance(r, APIRoute) and r.path == "/v1/chat/completions")
    ]

    @sgw_app.post("/v1/chat/completions")
    @limiter.limit("100/minute")
    async def chat_completion(
        chat_req: FullChatRequest,
        request: Request,
        x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    ):
        queue_time = time.perf_counter() - request.state.arrival_time
        req_id = x_request_id or str(uuid.uuid4())
        try:
            backend = _backends_mod.get_backend_instance(chat_req.model, CONFIG)
            reply = await backend.generate(chat_req)
        except Exception as e:
            reply = f"Gateway Error: {type(e).__name__}: {e}"
        if chat_req.stream:
            return StreamingResponse(
                generate_stream(req_id, reply),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        elapsed = time.perf_counter() - request.state.arrival_time
        return {
            "id": req_id,
            "choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(reply), "total_tokens": len(reply)},
            "metrics": {"queue_time": queue_time, "tftt": elapsed},
        }

    from simple_ai_gateway.main import main as gateway_main

    gateway_main()
