import modal
import subprocess
import time

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
        "hf_transfer",       # faster HF downloads
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


def download_model():
    """Downloads model weights into the volume at container boot time."""
    from huggingface_hub import snapshot_download
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors
    )


@app.cls(
    image=vllm_image,
    gpu="A10G",
    volumes={"/models": volume},
    # Scale to zero: container shuts down after 5 min of inactivity
    min_containers=0,           # scale to zero when idle
    max_containers=1,
    scaledown_window=300,
)
class VLLMServer:

    @modal.enter()
    def start_server(self):
        """Starts the vLLM server when the container boots."""
        import os

        # Download weights if not already cached in the volume
        if not os.path.exists(MODEL_DIR):
            print("Model not found in volume — downloading...")
            download_model()
            volume.commit()
        else:
            print("Model found in volume — skipping download.")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_DIR,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--dtype", "half",          # fp16 — fits comfortably on A10G
            "--served-model-name", SERVED_MODEL_NAME,  # so curl requests use the HF model name
        ]

        print(f"Starting vLLM server on port {VLLM_PORT}...")
        self.proc = subprocess.Popen(cmd)

        # Wait until the server is ready
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 120):
        """Polls the health endpoint until vLLM is ready."""
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

    @modal.exit()
    def stop_server(self):
        self.proc.terminate()
        self.proc.wait()

    @modal.web_server(port=VLLM_PORT, startup_timeout=180)
    def serve(self):
        """
        Exposes the vLLM HTTP server as a Modal web endpoint.
        vLLM is OpenAI-compatible:
          POST /v1/chat/completions
          POST /v1/completions
          GET  /v1/models
        """
        pass