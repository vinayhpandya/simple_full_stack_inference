"""
Modal deployment: Llama 3.1 8B Instruct
Inference engine: vLLM (via `vllm serve`)
KV cache manager: LMCache v1 (CPU RAM offload, 20GB pool)

Prerequisites:
  1. Install Modal CLI:         pip install modal
  2. Authenticate:              modal setup
  3. Create HF token secret:   modal secret create huggingface-secret HF_TOKEN=<your_token>
  4. Deploy:                    modal deploy deploy_llama.py

Scaling controls (set before deploying):
  MODAL_MIN_CONTAINERS=1  → keep one GPU warm at all times (lower latency, higher cost)
  MODAL_MIN_CONTAINERS=0  → scale to zero when idle (default; cold start ~1-2 min from volume)
  MODAL_SCALEDOWN_WINDOW=300 → seconds to wait before scaling down after last request

The deployed endpoint is OpenAI-API-compatible:
  https://<workspace>--llama31-openai.modal.run/v1/chat/completions
"""

import os
import subprocess
import time

import modal

# ---------------------------------------------------------------------------
# Scaling controls — configurable at deploy time via env vars, no code changes
# ---------------------------------------------------------------------------
_MIN_CONTAINERS = max(0, int(os.environ.get("MODAL_MIN_CONTAINERS", "0")))
_SCALEDOWN_WINDOW = max(60, int(os.environ.get("MODAL_SCALEDOWN_WINDOW", "300")))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_DIR = "/models/llama31-8b"
SERVED_MODEL_NAME = "llama3.1-8b-lmcache"
VLLM_PORT = 8000

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
# - vllm: latest stable, includes the v1 engine LMCacheConnectorV1 requires
# - lmcache: registers the vLLM KV connector at install time
# - hf_transfer: faster HuggingFace downloads into the volume
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "lmcache",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ---------------------------------------------------------------------------
# Volume — caches model weights so cold starts load from disk, not HuggingFace
# First deploy will download ~16GB; subsequent cold starts skip this entirely.
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)

app = modal.App("llama31-8b-lmcache", image=image)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_model() -> None:
    """Downloads Llama 3.1 8B weights into the persistent volume."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors
    )


# ---------------------------------------------------------------------------
# Serve function
# ---------------------------------------------------------------------------
# memory=51200  → 50 GB RAM: 20 GB LMCache CPU pool + ~16 GB weights + overhead
# gpu="A10G"    → 24 GB VRAM: fits fp16 8B model with ~8 GB left for GPU-side
#                 KV cache before LMCache starts offloading to CPU RAM.
# max_containers=1 → prevents accidental multi-replica GPU bill.
# startup_timeout=180 → Modal's built-in fix for the race condition: proxy waits
#                 up to 180s for the port to be ready before routing any traffic.
# label → stable URL that won't change between deploys.
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    memory=51200,
    volumes={"/models": volume},
    min_containers=1,
    max_containers=1,
    scaledown_window=_SCALEDOWN_WINDOW,
    timeout=60 * 30,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=180,
    label="llama31-openai",
)
def serve() -> None:
    # --- Model weights ------------------------------------------------------
    if not os.path.exists(MODEL_DIR):
        print("Model not found in volume — downloading (~16GB, first deploy only)...")
        download_model()
        volume.commit()
    else:
        print("Model found in volume — skipping download.")

    # --- LMCache configuration via env vars ---------------------------------
    # These are read by LMCache at startup when the vLLM process initialises
    # the connector. vLLM itself does not read these — they are LMCache-only.
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"  # required for v1 connector
    os.environ["LMCACHE_LOCAL_CPU"] = "True"          # enable CPU RAM offload backend
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "20.0" # hard cap in GB; LRU eviction when full
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"          # KV blocks transferred in 256-token chunks

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_DIR,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", SERVED_MODEL_NAME,

        # --- LMCache connector ----------------------------------------------
        # Hands KV cache management from vLLM to LMCacheConnectorV1.
        # Without this flag, all LMCACHE_* env vars above are set but
        # LMCache is never loaded — the model serves with no caching at all.
        # kv_role=kv_both: this instance handles both prefill and decode.
        "--kv-transfer-config",
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',

        # --- Required when using LMCacheConnectorV1 -------------------------
        # LMCache owns prefix caching — vLLM's built-in must be off.
        "--no-enable-prefix-caching",
        # Eager mode avoids CUDA graph conflicts with LMCache connector hooks.
        "--enforce-eager",

        # --- General serving config -----------------------------------------
        "--dtype", "half",
        "--max-model-len", "8192"
    ]

    print(f"Starting vLLM + LMCache on port {VLLM_PORT}...")
    subprocess.Popen(cmd)