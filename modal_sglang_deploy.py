"""
Modal deployment for TinyLlama using SGLang (OpenAI-compatible HTTP API).

Deploy:
  uv run modal deploy modal_sglang_deploy.py

Public URL (after deploy):
  https://<workspace>--tinyllama-sglang.modal.run/v1/chat/completions

Use the same `tinyllama` served model name in client requests as in modal_vllm_deploy.py.
Do not deploy both apps with conflicting resource needs on the same workspace label;
this app uses label `tinyllama-sglang` so it can run alongside the vLLM app (`tinyllama-openai`).
"""

import os
import subprocess
import time

import modal

# At `modal deploy` time: MODAL_MIN_CONTAINERS=1 keeps one GPU warm (faster requests, higher cost).
_MIN_CONTAINERS = max(0, int(os.environ.get("MODAL_MIN_CONTAINERS", "0")))
_SCALEDOWN_WINDOW = max(60, int(os.environ.get("MODAL_SCALEDOWN_WINDOW", "300")))
# First boot: model load + SGLang/Triton warmup often exceeds 3–5 minutes.
_READY_TIMEOUT_SEC = max(120, int(os.environ.get("SGLANG_READY_TIMEOUT_SEC", "720")))

# ---------------------------------------------------------------------------
# Constants — keep MODEL_ID / paths aligned with modal_vllm_deploy.py if you
# want the same weights layout (optional: point both at the same HF repo).
# ---------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = "/models/tinyllama"
SERVED_MODEL_NAME = "tinyllama"
SERVER_PORT = 8000

# ---------------------------------------------------------------------------
# Image — SGLang's official pre-built Docker image.
# All FlashInfer kernels are pre-compiled → no nvcc needed at runtime, no JIT crash.
# cu129 is backwards-compatible with Modal's A10G (CUDA 12.x) drivers.
# See https://hub.docker.com/r/lmsysorg/sglang/tags for available tags.
# ---------------------------------------------------------------------------
sglang_image = (
    modal.Image.from_registry(
        # lmsysorg/sglang ships its own Python + pre-compiled FlashInfer kernels.
        # Do NOT add add_python here: Modal's injected Python would become the default
        # `python` binary but would lack sglang, causing launch_server to silently crash.
        "lmsysorg/sglang:v0.5.8.post1-cu129-amd64",
    )
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Separate volume from vLLM so the two stacks do not contend on the same volume mount.
volume = modal.Volume.from_name("sglang-model-cache", create_if_missing=True)

app = modal.App("tinyllama-sglang")


def download_model() -> None:
    """Download model weights into the volume."""
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],
    )


def wait_for_server(timeout: int) -> None:
    """Wait until SGLang reports ready (HTTP 200 on /health only).

    Do not use /v1/models as readiness: it can return 200 before the scheduler is
    actually able to run a batch (then the worker crashes and Modal restarts the container).
    """
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{SERVER_PORT}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                if r.getcode() == 200:
                    print("SGLang /health returned 200 — ready for traffic.")
                    return
        except urllib.error.HTTPError as e:
            # 503 while weights/scheduler still starting
            if e.code not in (503, 502):
                raise
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        remaining = int(deadline - time.time())
        print(f"Waiting for SGLang /health == 200 (~{remaining}s left)...")
        time.sleep(5)

    raise TimeoutError(
        f"SGLang /health did not return 200 within {timeout}s. "
        "Increase SGLANG_READY_TIMEOUT_SEC or check Modal logs for launch_server errors."
    )


@app.function(
    image=sglang_image,
    gpu="A10G",
    volumes={"/models": volume},
    min_containers=_MIN_CONTAINERS,
    max_containers=1,
    scaledown_window=_SCALEDOWN_WINDOW,
    # Covers the SGLang startup time. After serve() returns, Modal keeps the container alive
    # via @modal.web_server, so the timeout only needs to be longer than the startup path.
    timeout=1800,
)
@modal.web_server(
    port=SERVER_PORT,
    # Must cover worst-case cold start (download is rare; compile/graph capture is not).
    startup_timeout=float(os.environ.get("MODAL_WEB_STARTUP_TIMEOUT_SEC", "900")),
    label="tinyllama-sglang",
)
def serve() -> None:
    """
    Start SGLang's OpenAI-compatible server and expose it on SERVER_PORT.

    Endpoints include POST /v1/chat/completions and GET /health.
    """
    import os

    if not os.path.exists(MODEL_DIR):
        print("Model not found in volume — downloading...")
        download_model()
        volume.commit()
    else:
        print("Model found in volume — skipping download.")

    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_DIR,
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--dtype",
        "half",
        "--served-model-name",
        SERVED_MODEL_NAME,
        # FlashInfer kernels are pre-compiled in lmsysorg/sglang → no JIT, no nvcc needed.
        # Remove the two lines below if you switch back to debian_slim (they prevent the crash).
        # "--attention-backend", "triton",
        # "--sampling-backend", "pytorch",
        # Disable CUDA graph capture for faster cold boot; remove for max throughput.
        "--disable-cuda-graph",
        # Raise SGLang's idle watchdog well above any realistic idle gap between requests.
        # Default is 300s; with min_containers=1 a warm container may sit idle for longer.
        "--watchdog-timeout",
        "3600",
    ]

    print(
        f"Starting SGLang on port {SERVER_PORT} (ready timeout {_READY_TIMEOUT_SEC}s, "
        f"Modal web startup {os.environ.get('MODAL_WEB_STARTUP_TIMEOUT_SEC', '900')}s)..."
    )
    proc = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=None,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    wait_for_server(_READY_TIMEOUT_SEC)
    # serve() returns here. Modal's @web_server decorator keeps the container alive and
    # proxies all HTTP traffic to SGLang on SERVER_PORT. Do NOT call proc.wait() here —
    # blocking serve() prevents Modal from routing requests to the port.
