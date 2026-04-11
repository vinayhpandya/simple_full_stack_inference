import modal
import os

# ---------------------------------------------------------------------------
# Image — install vllm + lmcache + httpx
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
   "vllm==0.7.3",
    "huggingface_hub",
    "httpx"
    )
    .pip_install("transformers==4.49.0")
    .add_local_file("lmcache_config.yaml", "/root/lmcache_config.yaml")
)

app = modal.App("qwen2.5-7b-lmcache")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_DIR = "/model"

# ---------------------------------------------------------------------------
# Volume — cache model weights so containers don't re-download on every start
# ---------------------------------------------------------------------------
model_volume = modal.Volume.from_name("qwen2.5-7b-weights", create_if_missing=True)

# ---------------------------------------------------------------------------
# Download model weights once into the volume
# ---------------------------------------------------------------------------
@app.function(
    image=vllm_image,
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],  # fix 2: unified secret name
    timeout=3600,
)
def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
    )
    model_volume.commit()
    print(f"Model downloaded to {MODEL_DIR}")


# ---------------------------------------------------------------------------
# Serving class
# ---------------------------------------------------------------------------
@app.cls(
    image=vllm_image,
    gpu="A10G",                          # 24GB VRAM — plenty for 7B
    min_containers=1,                    # always-warm, no cold starts
    max_containers=4,                    # autoscale up to 4 replicas
    scaledown_window=300,                # wait 5 min of idle before scaling down
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    timeout=600,
)
@modal.concurrent(max_inputs=64)
class QwenServer:
    @modal.enter()
    def start_engine(self):
        import subprocess, sys, time, httpx

        # LMCache hooks into vllm via env vars
        os.environ["LMCACHE_CONFIG_FILE"] = "/root/lmcache_config.yaml"

        # fix 3: use the standard vllm entrypoint — lmcache patches in via env vars
        self.proc = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_ID,
                "--download-dir", MODEL_DIR,
                "--served-model-name", "qwen2.5-7b",
                "--dtype", "bfloat16",
                "--max-model-len", "8192",
                "--max-num-batched-tokens", "8192",
                "--enable-prefix-caching",
                "--gpu-memory-utilization", "0.88",
                "--port", "8000",
                "--trust-remote-code",
            ]
        )

        # Wait until vllm is ready
        for _ in range(120):
            try:
                r = httpx.get("http://127.0.0.1:8000/health", timeout=2)
                if r.status_code == 200:
                    print("vllm ready")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("vllm did not start in time")

    @modal.exit()
    def stop_engine(self):
        self.proc.terminate()
        self.proc.wait()

    @modal.web_server(port=8000, startup_timeout=300)
    def serve(self):
        """Exposes the vllm OpenAI-compatible server directly."""
        pass


# ---------------------------------------------------------------------------
# Entrypoint — run `modal run modal_deploy.py` to download weights first,
# then `modal deploy modal_deploy.py` to deploy
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    download_model.remote()
    print("Weights ready. Deploy with: modal deploy modal_deploy.py")