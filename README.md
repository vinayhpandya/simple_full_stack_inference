# Simple Full Stack Inference

A production-grade AI inference platform demonstrating a full-stack LLM serving architecture with multiple inference backends, load balancing, monitoring, and metrics collection.

## Architecture

```
                 External Requests
                         │
                         ▼
              ┌─────────────────────┐
              │  Nginx Load Balancer │
              │      (Port 80)       │
              │  - SSE streaming     │
              │  - long timeouts     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    API Gateway       │
              │     (Port 8080)      │
              │  - Metrics endpoint  │
              │  - Backend routing   │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Prometheus  │  │  Modal vLLM  │  │ Anyscale DeepSeek│
│ (Port 9090) │◄─│  / SGLang    │  │  Ray Serve +     │
│   Metrics   │  │  TinyLlama   │  │  vLLM DeepSeek   │
└──────┬──────┘  └──────────────┘  │  V2-Lite-Chat    │
       │                           └──────────────────┘
       ▼
┌─────────────┐
│   Grafana   │
│ (Port 3000) │
│  Dashboards │
└─────────────┘
```

## Components

| Component | Description | Port |
|-----------|-------------|------|
| **API Gateway** | Request routing, backend selection, and metrics | 8080 |
| **Nginx** | Load balancer with SSE streaming support | 80 |
| **Modal vLLM / SGLang** | TinyLlama-1.1B on Modal GPU (auto-scaling) | — (HTTPS) |
| **Anyscale DeepSeek** | DeepSeek-V2-Lite-Chat via Ray Serve + vLLM on Anyscale | — (HTTPS) |
| **Prometheus** | Metrics collection and storage | 9090 |
| **Grafana** | Metrics visualization and dashboards | 3000 |

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker Desktop (or another Docker engine) running, with Compose available
- [Modal](https://modal.com) account (for GPU inference via vLLM / SGLang)
- [Anyscale](https://www.anyscale.com) account (for DeepSeek MoE inference — optional)
- **Nginx** (optional, for port 80): on macOS, `brew install nginx`; binding to port 80 usually requires `sudo`

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd simple_full_stack_inference
```

### 2. Install Python Dependencies

```bash
uv sync
```

### 3. Configure Modal

The Modal CLI is installed with this project. Run it via `uv` so it uses the project virtualenv:

```bash
# Authenticate (opens browser / token flow)
uv run modal token new
```

### 4. Deploy inference to Modal (vLLM or SGLang)

Pick **one** engine (or deploy both to different Modal apps / labels and point the gateway at whichever you use).

#### Option A — vLLM (default)

```bash
uv run modal deploy modal_vllm_deploy.py
```

- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **GPU**: NVIDIA A10G
- **Features**: Auto-scaling, scale-to-zero after 5 min idle
- **Web endpoint**: `https://<your-workspace>--tinyllama-openai.modal.run` (`label` in `modal_vllm_deploy.py`)

In **`gateway_config.yaml`**, set `backends.modal_vllm.url` to:

`https://<your-workspace>--tinyllama-openai.modal.run/v1/chat/completions`

#### Option B — SGLang

```bash
uv run modal deploy modal_sglang_deploy.py
```

Same TinyLlama weights and **`tinyllama`** served name, but the server is [SGLang](https://github.com/sgl-project/sglang) (`python -m sglang.launch_server`). First image build can take longer than vLLM because of `sglang[all]`. The deploy script uses **`--attention-backend triton`** and **`--sampling-backend pytorch`** because the default FlashInfer path **JIT-compiles CUDA with `nvcc`**, which is **not** installed on `debian_slim` (you would see `Could not find nvcc` and a Modal restart loop). It waits for **`GET /health` → 200** only (not `/v1/models`). Tunables: **`SGLANG_READY_TIMEOUT_SEC`** (default **720**), **`MODAL_WEB_STARTUP_TIMEOUT_SEC`** (default **900**), **`--disable-cuda-graph`** in `modal_sglang_deploy.py` if you want graphs back on a **CUDA devel** base image.

- **Web endpoint**: `https://<your-workspace>--tinyllama-sglang.modal.run` (`label` in `modal_sglang_deploy.py`)
- **Volume**: `sglang-model-cache` (separate from vLLM’s cache)

Point **`gateway_config.yaml`** at SGLang by setting `backends.modal_vllm.url` (or add another backend key) to:

`https://<your-workspace>--tinyllama-sglang.modal.run/v1/chat/completions`

Keep **`type: vllm`** (or **`type: remote`**) for that backend — the PyPI gateway has no **`sglang`** type; SGLang is still an OpenAI-compatible HTTP proxy target.

#### After either deploy

Copy the exact HTTPS origin from the CLI or [Modal dashboard](https://modal.com), append **`/v1/chat/completions`**, update **`gateway_config.yaml`**, then restart **`uv run simple-ai-gateway`**.

---

### 4b. (Optional) Deploy DeepSeek MoE to Anyscale

This deploys **DeepSeek-V2-Lite-Chat** (16B MoE / 2.4B active params) via Ray Serve + vLLM on an Anyscale-managed A10G GPU.

#### Step 1 — Install the Anyscale CLI

```bash
uv add anyscale
```

#### Step 2 — Create an Anyscale account and API key

1. Sign up at [anyscale.com](https://www.anyscale.com) (free trial available).
2. Go to **Settings → API Keys** and create a new key.
3. Authenticate the CLI (interactive prompt — paste the key when asked):

```bash
uv run anyscale auth set
```

#### Step 3 — Create a cloud

In the [Anyscale console](https://console.anyscale.com), navigate to **Clouds** and create a cloud (the default name is **`Anyscale Cloud`**). Anyscale will provision AWS/GCP resources in its own managed account — no AWS credentials needed for the managed option.

If you renamed your cloud, update `compute_config.cloud` in `anyscale_service.yaml` to match.

#### Step 4 — Accept the DeepSeek model terms and store your HF token

Accept the HuggingFace gated-model terms at:  
<https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat>

Then open **`anyscale_service.yaml`** and fill in your HuggingFace token:

```yaml
env_vars:
  HF_TOKEN: "hf_your_token_here"   # ← paste here; do not commit this file
  ...
```

> **Important:** `anyscale_service.yaml` is already in `.gitignore`. Never commit it with a real token.

#### Step 5 — Deploy the service

```bash
uv run anyscale service deploy -f anyscale_service.yaml
```

The first deploy takes **10–20 minutes** (GPU provisioning + model download). You will see:

```
(anyscale) Starting new service 'deepseek-moe'.
(anyscale) Service 'deepseek-moe' is now running.
(anyscale) Base URL: https://deepseek-moe-<hash>.cld_<id>.s.anyscaleuserdata.com
```

#### Step 6 — Update the gateway

Copy the **Base URL** from the deploy output (or from [console.anyscale.com/services](https://console.anyscale.com/services) → `deepseek-moe` → **Base URL**) and update `gateway_config.yaml`:

```yaml
backends:
  anyscale_deepseek:
    type: vllm
    url: https://deepseek-moe-<hash>.cld_<id>.s.anyscaleuserdata.com/v1/chat/completions
```

To make DeepSeek the default backend, also change:

```yaml
default_backend: anyscale_deepseek
```

Restart the gateway after any config change:

```bash
uv run simple-ai-gateway
```

#### Managing the Anyscale service (cost control)

Unlike Modal, Anyscale Ray Serve services **stay running (and billing) until you explicitly terminate them**. A `g5.2xlarge` (A10G) costs roughly **$1–2/hour** while the service is up.

**Terminate** when not in use:

```bash
uv run anyscale service terminate --name deepseek-moe
```

**Redeploy** when you need it again:

```bash
uv run anyscale service deploy -f anyscale_service.yaml
```

Check remaining credits / current spend at [console.anyscale.com/v2/billing](https://console.anyscale.com/v2/billing).

### 5. Start the Monitoring Stack

Docker must be running (`docker info` should succeed). From the repo root:

```bash
cd monitoring
docker compose up -d
```

If your machine only has the older standalone binary, use `docker-compose up -d` instead.

This starts:
- **Prometheus** at http://localhost:9090
- **Grafana** at http://localhost:3000 (admin/admin)

### 6. Start API gateway

From the **repository root** (not `monitoring/`):

```bash
uv run simple-ai-gateway
```

This loads **`gateway_config.yaml`** by default (via `gateway_launcher.py`) and proxies to Modal vLLM. The first request after scale-to-zero can take **minutes**; upstream timeout defaults to **600s** (override with `SIMPLE_AI_GATEWAY_UPSTREAM_TIMEOUT` if needed).

If port **8080** is already in use, stop the other process or set `PORT=8081` and point Nginx at that port in `nginx-lb.conf`.

### 7. Start nginx reverse proxy

`nginx-lb.conf` listens on **port 80** (HTTP). On macOS you usually need **sudo**:

```bash
cd /path/to/simple_full_stack_inference
sudo nginx -c "$(pwd)/nginx-lb.conf"
```

Stop later with `sudo nginx -s stop` (or `sudo nginx -s quit` for a graceful shutdown).

If `nginx` is not found, install it (e.g. `brew install nginx`). Avoid running Homebrew’s default `brew services start nginx` at the same time if it is also bound to **8080**, which would conflict with the API gateway.

## Usage

### API Endpoints (OpenAI Compatible)

The vLLM server exposes OpenAI-compatible endpoints:

Use **`http://`** (not `https://`) for local Nginx on port 80. The **`model`** field must match vLLM’s served name: **`tinyllama`** (see `SERVED_MODEL_NAME` in `modal_vllm_deploy.py`).

**Chat completions (via Nginx on port 80)**

```bash
curl -s -X POST "http://localhost/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

**Chat completions (direct to API gateway on port 8080)**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### DeepSeek-V2-Lite-Chat (Anyscale MoE backend)

Send requests to the `anyscale_deepseek` backend using the `X-Backend` header (leaves `modal_vllm` as default) or by setting `default_backend: anyscale_deepseek` in `gateway_config.yaml`.

**Via Nginx (port 80) — targeting DeepSeek explicitly:**

```bash
curl -s -X POST "http://localhost/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Backend: anyscale_deepseek" \
  -d '{
    "model": "deepseek",
    "messages": [
      {"role": "user", "content": "Explain mixture-of-experts in one sentence."}
    ],
    "max_tokens": 60
  }'
```

**Direct to API gateway (port 8080):**

```bash
curl -s -X POST "http://127.0.0.1:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Backend: anyscale_deepseek" \
  -d '{
    "model": "deepseek",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

The `model` field must match `DEEPSEEK_SERVED_NAME` in `anyscale_service.yaml` (default: **`deepseek`**). Anyscale keeps the service warm as long as it is running — no cold-start penalty after the first deploy.

### Streaming responses

The stack supports SSE streaming; use `-N` with `curl` so output is not buffered:

```bash
curl -N -X POST "http://localhost/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

### Why `curl` through localhost can take minutes

Nothing is “wrong” with Nginx or the gateway if the **first** request after idle is **very slow**. Most of the time is **Modal + the inference engine**, not your laptop.

1. **Scale to zero** — With `min_containers=0` (default), Modal may **cold-start a GPU**, load weights from volume, and boot vLLM/SGLang. That alone is often **1–5+ minutes** for the first call.
2. **Modal web hook limit (~150s per HTTP leg)** — If work is not done within ~150s, Modal returns **303** and the client must **GET** the result URL; that can repeat. Your gateway handles this; it still adds **wall-clock time** and connection churn. See [Modal webhook timeouts](https://modal.com/docs/guide/webhook-timeouts).
3. **SGLang** — First request can pay extra **JIT/compile** cost on top of model load.
4. **After the container is warm** — The **second** request in a short window is usually **much faster** (often seconds for TinyLlama).

**To prioritize latency over cost**, redeploy with a **warm** replica (one GPU stays allocated while idle):

```bash
MODAL_MIN_CONTAINERS=1 uv run modal deploy modal_vllm_deploy.py
# or for SGLang:
MODAL_MIN_CONTAINERS=1 uv run modal deploy modal_sglang_deploy.py
```

Optional: keep idle workers longer **without** `min_containers=1` by raising idle lifetime (seconds):

```bash
MODAL_SCALEDOWN_WINDOW=900 MODAL_MIN_CONTAINERS=0 uv run modal deploy modal_sglang_deploy.py
```

Environment variables are read when **`modal deploy`** runs (they are baked into the deployed Function config).

## Monitoring

### Grafana Dashboard

Access Grafana at http://localhost:3000 (default: admin/admin)

The pre-configured dashboard includes:
- **Gateway Metrics**: Latency, request counts, rate limits
- **vLLM Metrics**:
  - End-to-end latency
  - Time to first token (TTFT)
  - Inter-token latency
  - Token throughput
  - Prompt token recomputation

### Prometheus Targets

Prometheus is configured to scrape (see `monitoring/prometheus.yml`):
1. **API Gateway** — `host.docker.internal:8080/metrics` every 15s (`gateway_launcher.py` adds `/metrics` and HTTP metrics for Grafana.)
2. **vLLM on Modal** — HTTPS metrics path on your Modal host, if reachable from Prometheus

View targets at http://localhost:9090/targets


### Nginx Load Balancer (`nginx-lb.conf`)

Key settings for LLM inference:
- **Buffering disabled**: Required for SSE streaming
- **Long proxy timeouts (900s)**: Matches Modal cold start + webhook 303 legs + generation (see `nginx-lb.conf`)
- **Keepalive connections**: Efficient connection reuse

## Troubleshooting

**`Backend Error (...)` from the API gateway (curl to `http://localhost/...`)**

1. **Restart the gateway** from the repo root after pulling changes: `uv run simple-ai-gateway` (config and `gateway_launcher.py` patches load at startup).
2. **`gateway_config.yaml`** — `modal_vllm.url` must be the full chat URL, e.g. `https://<workspace>--tinyllama-sglang.modal.run/v1/chat/completions` (SGLang) or `...--tinyllama-openai...` (vLLM).
3. **`model` in JSON** — Must match what Modal serves (`tinyllama` in this repo). If you use `"model": "modal_vllm"` only for routing, the gateway would forward that name upstream and SGLang/vLLM can return **400**; either send **`"model": "tinyllama"`** in the client body or set **`SIMPLE_AI_GATEWAY_UPSTREAM_MODEL=tinyllama`** when starting the gateway to rewrite the upstream payload.
4. **Modal 303 / long webhooks** — Modal may return **303** after **~150s** per HTTP leg and a `Location` with **`__modal_function_call_id`**. That URL must be loaded with **GET** (no JSON body); it blocks until inference finishes. A second **POST** there returns `modal-http: bad redirect method`. See [Modal webhook timeouts](https://modal.com/docs/guide/webhook-timeouts). The gateway handles this; restart it after upgrading `gateway_launcher.py`.
5. **Long first request** — Cold start can take **minutes**; optional `SIMPLE_AI_GATEWAY_UPSTREAM_TIMEOUT=900`.
6. **Proxies** — The gateway uses `trust_env=False` for upstream Modal calls so `HTTP_PROXY` does not break TLS to `*.modal.run`. If you need a proxy for Modal, we’d have to wire it explicitly.
7. Read the text after **`—`** in the error string; it is the upstream **response body** (SGLang/Modal validation message).

**vLLM server not responding**
- Check Modal dashboard for deployment status
- Verify the model is downloaded (first cold start takes longer)
- Logs: `uv run modal app logs` (pick your app / follow CLI prompts)

**`command not found: modal`**
- Use `uv run modal ...` from the repo root after `uv sync`

**Anyscale deployment issues**

- **`No such command 'secret'`** — This CLI version has no `secret` command; put `HF_TOKEN` directly in `anyscale_service.yaml` under `env_vars` and keep the file out of git (it is already in `.gitignore`).
- **`Cluster environment with image URI … not found`** — Remove the `image_uri` field from `anyscale_service.yaml`; Anyscale uses its default managed Ray image and installs packages from `requirements`.
- **`ServiceConfig.__init__() got an unexpected keyword argument 'ray_serve_config'`** — The installed CLI uses the new schema: use `applications` (list) instead of `ray_serve_config`, `head_node` instead of `head_node_type`, and `worker_nodes` instead of `worker_node_types`.
- **Service URL not visible in `anyscale service list`** — Run `uv run anyscale service status --name deepseek-moe` or check the [Anyscale console](https://console.anyscale.com/services) for the **Base URL**.
- **`model` field mismatch** — The `model` sent in the request body must match `DEEPSEEK_SERVED_NAME` in `anyscale_service.yaml` (default: `deepseek`). Sending a different name causes a 404 from vLLM.
- **Logs** — View Ray Serve / vLLM worker logs in the [Anyscale console](https://console.anyscale.com/services) → `deepseek-moe` → **Logs** tab.

**`nginx: command not found`**
- Install Nginx (e.g. `brew install nginx` on macOS)

**`Cannot connect to the Docker daemon` / compose errors**
- Start Docker Desktop (or your engine) and wait until it is fully running; confirm with `docker info`

**Prometheus not scraping metrics**
- Verify targets at http://localhost:9090/targets
- Check if API gateway is running on port 8080
- Ensure `host.docker.internal` resolves correctly

**Grafana dashboard empty**
- Verify Prometheus datasource is configured
- Check that Prometheus is receiving metrics
- Import the dashboard from `monitoring/grafana_dashboards/`
