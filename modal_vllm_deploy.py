import os
import subprocess
import time

import modal

# At `modal deploy` time: MODAL_MIN_CONTAINERS=1 keeps one GPU replica warm (lower latency, higher cost).
# Default 0 = scale to zero when idle (first request after idle is often 1–5+ minutes).
_MIN_CONTAINERS = max(0, int(os.environ.get("MODAL_MIN_CONTAINERS", "0")))
# Seconds to keep containers up after traffic stops (only when min_containers=0).
_SCALEDOWN_WINDOW = max(60, int(os.environ.get("MODAL_SCALEDOWN_WINDOW", "300")))

# ---------------------------------------------------------------------------
# Constants — change MODEL_ID to swap models
# ---------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR = "/models/tinyllama"
SERVED_MODEL_NAME = "tinyllama"
VLLM_PORT = 8000

# ---------------------------------------------------------------------------
# Image — installs vLLM and its dependencies
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "huggingface_hub",
        "hf_transfer",  # faster HF downloads
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Volume — caches model weights so cold starts don't re-download
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("tinyllama-vllm")


def download_model() -> None:
    """Downloads model weights into the volume at container boot time."""
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors
    )


def wait_for_vllm(timeout: int = 120) -> None:
    import urllib.request

    url = f"http://localhost:{VLLM_PORT}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            print("vLLM server is ready.")
            return
        except Exception:
            print("Waiting for vLLM server...")
            time.sleep(3)

    raise TimeoutError("vLLM server did not start in time.")


# Use @app.function + @modal.web_server (not @app.cls) so Modal proxies paths
# correctly to vLLM — same pattern as https://modal.com/docs/examples/vllm_inference
@app.function(
    image=vllm_image,
    gpu="A10G",
    volumes={"/models": volume},
    min_containers=_MIN_CONTAINERS,
    max_containers=1,
    scaledown_window=_SCALEDOWN_WINDOW,
    timeout=60 * 30,
)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=180,
    # Stable subdomain: https://<workspace>--tinyllama-openai.modal.run
    # (Without `label`, Modal auto-picks a suffix; guessing it causes 404s.)
    label="tinyllama-openai",
)
def serve() -> None:
    """
    Starts vLLM and exposes it at the deployed URL (OpenAI-compatible API).

    After `modal deploy`, use:
      https://<workspace>--tinyllama-openai.modal.run/v1/chat/completions
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
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_DIR,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "half",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]

    print(f"Starting vLLM server on port {VLLM_PORT}...")
    subprocess.Popen(cmd)
    wait_for_vllm()
