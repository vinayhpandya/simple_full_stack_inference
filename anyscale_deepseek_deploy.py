"""
Anyscale deployment for DeepSeek-V2-Lite-Chat via Ray Serve + vLLM.

Deploy:
  anyscale service deploy anyscale_service.yaml

The service exposes an OpenAI-compatible HTTP API at the URL printed by
`anyscale service deploy`. Point gateway_config.yaml at that URL.

Environment variables (set in anyscale_service.yaml or via `anyscale secret`):
  DEEPSEEK_MODEL_ID     — HuggingFace repo (default: deepseek-ai/DeepSeek-V2-Lite-Chat)
  DEEPSEEK_SERVED_NAME  — model name clients send in requests (default: deepseek)
  TENSOR_PARALLEL_SIZE  — number of GPUs per replica (default: 1)
  VLLM_READY_TIMEOUT    — seconds to wait for vLLM /health (default: 600)
  HF_TOKEN              — HuggingFace token for gated-model access
"""

import os
import subprocess
import time
import urllib.error
import urllib.request

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response
from ray import serve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-V2-Lite-Chat")
SERVED_MODEL_NAME = os.environ.get("DEEPSEEK_SERVED_NAME", "deepseek")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
READY_TIMEOUT = int(os.environ.get("VLLM_READY_TIMEOUT", "600"))
SERVER_PORT = 8000

# ---------------------------------------------------------------------------
# FastAPI app — used as the Ray Serve ingress
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(title="DeepSeek-V2-Lite-Chat")


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_gpus": TENSOR_PARALLEL_SIZE},
    # Queue at most 8 in-flight requests per replica before Ray Serve applies
    # back-pressure. Raise this if you want higher concurrency per GPU.
    max_ongoing_requests=8,
)
@serve.ingress(fastapi_app)
class DeepSeekDeployment:
    """Ray Serve actor that starts vLLM as a subprocess and proxies HTTP to it."""

    def __init__(self) -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_ID,
            "--served-model-name", SERVED_MODEL_NAME,
            "--host", "127.0.0.1",
            "--port", str(SERVER_PORT),
            "--dtype", "bfloat16",
            "--trust-remote-code",
            "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
            # Disable CUDA graph capture for a faster first start; remove for
            # maximum throughput once the deployment is stable.
            "--enforce-eager",
        ]

        print(
            f"Starting vLLM for {MODEL_ID} on port {SERVER_PORT} "
            f"(tp={TENSOR_PARALLEL_SIZE}, ready timeout {READY_TIMEOUT}s)..."
        )
        self._proc = subprocess.Popen(cmd, env=env)
        self._wait_for_ready()

        # Keep one persistent httpx client for the lifetime of this actor.
        self._client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{SERVER_PORT}",
            timeout=httpx.Timeout(600.0, connect=10.0),
        )

    def _wait_for_ready(self) -> None:
        """Poll GET /health until vLLM returns 200 or the timeout elapses."""
        url = f"http://127.0.0.1:{SERVER_PORT}/health"
        deadline = time.time() + READY_TIMEOUT

        while time.time() < deadline:
            # If the subprocess crashed, surface the error immediately rather
            # than waiting for the full timeout.
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM process exited with code {self._proc.returncode} "
                    "before becoming healthy."
                )
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    if r.getcode() == 200:
                        print("vLLM /health returned 200 — ready for traffic.")
                        return
            except urllib.error.HTTPError as e:
                if e.code not in (503, 502):
                    raise
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            remaining = int(deadline - time.time())
            print(f"Waiting for vLLM /health == 200 (~{remaining}s left)...")
            time.sleep(5)

        raise TimeoutError(
            f"vLLM /health did not return 200 within {READY_TIMEOUT}s. "
            "Check Anyscale worker logs for model-load errors."
        )

    # Catch-all: proxy every request path and method to the local vLLM server.
    @fastapi_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(self, request: Request, path: str) -> Response:
        body = await request.body()
        # Strip the host header so httpx doesn't send it to 127.0.0.1 (causes
        # vLLM to reject it as a bad Host header on some versions).
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        upstream = await self._client.request(
            method=request.method,
            url=f"/{path}",
            params=dict(request.query_params),
            headers=headers,
            content=body,
        )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=dict(upstream.headers),
        )


# ---------------------------------------------------------------------------
# Bind — this is what anyscale_service.yaml references as the import path
# ---------------------------------------------------------------------------

deployment = DeepSeekDeployment.bind()
