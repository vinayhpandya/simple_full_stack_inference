import os
from ray.serve.llm import LLMConfig, build_pd_openai_app

MODEL_ID = os.environ.get("MODEL_ID", "deepseek-ai/DeepSeek-V2-Lite")

# ── Prefill config ────────────────────────────────────────────────────────────
prefill_config = LLMConfig(
    model_loading_config={
        "model_id": MODEL_ID,
    },
    deployment_config={
        "num_replicas": 1,
        "max_ongoing_requests": 64,
        # ❌ removed: request_router_config — not a valid deployment() kwarg
    },
    engine_kwargs={
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": True,
        "trust_remote_code": True,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
        },
    },
)

# ── Decode config ─────────────────────────────────────────────────────────────
decode_config = LLMConfig(
    model_loading_config={
        "model_id": MODEL_ID,
    },
    deployment_config={
        "num_replicas": 1,
        "max_ongoing_requests": 256,
    },
    engine_kwargs={
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "enable_chunked_prefill": False,
        "trust_remote_code": True,
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
        },
    },
)

# ── Build the P/D app ─────────────────────────────────────────────────────────
app = build_pd_openai_app({
    "prefill_config": prefill_config,
    "decode_config": decode_config,
})