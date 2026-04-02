# Simple Full Stack Inference

A production-grade AI inference platform demonstrating a full-stack LLM serving architecture with load balancing, monitoring, and metrics collection.

## Architecture

```
                 External Requests
                         │
                         ▼
              ┌─────────────────────┐
              │  Nginx Load Balancer │
              │      (Port 80)       │
              │  - SSE streaming     │
              │  - 180s timeouts     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    API Gateway       │
              │     (Port 8080)      │
              │  - Metrics endpoint  │
              └──────────┬──────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│   Prometheus    │            │   vLLM Server   │
│   (Port 9090)   │◄───────────│   (Port 8000)   │
│  Metrics Store  │   scrape   │  Modal-hosted   │
└────────┬────────┘            │  TinyLlama 1.1B │
         │                     └─────────────────┘
         ▼
┌─────────────────┐
│     Grafana     │
│   (Port 3000)   │
│   Dashboards    │
└─────────────────┘
```

## Components

| Component | Description | Port |
|-----------|-------------|------|
| **vLLM Server** | LLM inference server running TinyLlama-1.1B on Modal | 8000 |
| **API Gateway** | Request routing and metrics exposure | 8080 |
| **Nginx** | Load balancer with SSE streaming support | 80 |
| **Prometheus** | Metrics collection and storage | 9090 |
| **Grafana** | Metrics visualization and dashboards | 3000 |

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose
- [Modal](https://modal.com) account (for GPU inference)

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

```bash
# Install Modal CLI if not already installed
pip install modal

# Authenticate with Modal
modal token new
```

### 4. Deploy vLLM Server to Modal

```bash
modal deploy modal_vllm_deploy.py
```

This will deploy a vLLM server with:
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **GPU**: NVIDIA A10G
- **Features**: Auto-scaling, scale-to-zero after 5 min idle

### 5. Start the Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

This starts:
- **Prometheus** at http://localhost:9090
- **Grafana** at http://localhost:3000 (admin/admin)

### 6. Start api gateway
```bash
uv run simple-ai-gateway
```

### 7. Start nginx reverse proxy
```bash
nginx -c $(pwd)/nginx-lb.conf
```

## Usage

### API Endpoints (OpenAI Compatible)

The vLLM server exposes OpenAI-compatible endpoints:

**Chat Completions**
```bash
curl -X POST "https://localhost:80/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "modal_vllm",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```


### Streaming Responses

The server supports SSE streaming for real-time token generation:

```bash
curl -X POST "https://localhost:80/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "modal_vllm",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

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

Prometheus scrapes metrics from:
1. **API Gateway** (`localhost:8080/metrics`) - every 15s
2. **vLLM Server** (Modal URL) - every 15s

View targets at http://localhost:9090/targets


### Nginx Load Balancer (`nginx-lb.conf`)

Key settings for LLM inference:
- **Buffering disabled**: Required for SSE streaming
- **180s timeouts**: Accommodates long-running inference requests
- **Keepalive connections**: Efficient connection reuse

## Troubleshooting

**vLLM server not responding**
- Check Modal dashboard for deployment status
- Verify the model is downloaded (first cold start takes longer)
- Check logs: `modal app logs`

**Prometheus not scraping metrics**
- Verify targets at http://localhost:9090/targets
- Check if API gateway is running on port 8080
- Ensure `host.docker.internal` resolves correctly

**Grafana dashboard empty**
- Verify Prometheus datasource is configured
- Check that Prometheus is receiving metrics
- Import the dashboard from `monitoring/grafana/grafana_dashboards/`
