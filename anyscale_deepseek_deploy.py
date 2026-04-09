"""
Anyscale deployment for DeepSeek-V2-Lite-Chat via Ray Serve + vLLM.

Architecture: Ray Serve owns the HTTP layer and orchestrates inference directly.
vLLM's AsyncLLMEngine runs *inside* the Serve actor as a Python object — no
subprocess, no httpx proxy, no port binding. Ray Serve sees the in-flight
request count and can scale replicas with full visibility.

                 ┌─────────────────────────────────┐
                 │       Ray Serve actor            │
                 │  ┌───────────────────────────┐  │
  HTTP request ──►  │  FastAPI /v1/chat/...     │  │
                 │  └────────────┬──────────────┘  │
                 │               │  Python call     │
                 │  ┌────────────▼──────────────┐  │
                 │  │   AsyncLLMEngine (vLLM)   │  │
                 │  │   GPU memory managed by   │  │
                 │  │   Ray resource scheduler  │  │
                 │  └───────────────────────────┘  │
                 └─────────────────────────────────┘

Deploy:
  uv run anyscale service deploy -f anyscale_service.yaml

Environment variables (set in anyscale_service.yaml env_vars):
  DEEPSEEK_MODEL_ID     — HuggingFace repo (default: deepseek-ai/DeepSeek-V2-Lite-Chat)
  DEEPSEEK_SERVED_NAME  — model name clients send in requests (default: deepseek)
  TENSOR_PARALLEL_SIZE  — number of GPUs per replica (default: 1)
  HF_TOKEN              — HuggingFace token for gated-model access
"""

import json
import os
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from ray import serve

# ---------------------------------------------------------------------------
# Configuration (read once at import time so Ray can serialise the class
# without needing vLLM installed on the head node)
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-V2-Lite-Chat")
SERVED_MODEL_NAME = os.environ.get("DEEPSEEK_SERVED_NAME", "deepseek")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))

# ---------------------------------------------------------------------------
# Request / response schema (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class _Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[_Message]
    model: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None
    n: int = 1

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
    # Ray Serve applies back-pressure once this many requests are in flight
    # per replica. Raise if you need higher concurrency per GPU.
    max_ongoing_requests=8,
)
@serve.ingress(fastapi_app)
class DeepSeekDeployment:
    """
    Ray Serve actor that loads vLLM's AsyncLLMEngine directly.

    Unlike the subprocess approach, Ray Serve has full visibility into GPU
    memory and in-flight requests, enabling accurate autoscaling decisions.
    """

    def __init__(self) -> None:
        # Defer vLLM imports to the actor's __init__ so Ray can pickle the
        # class definition without requiring vLLM on the head node.
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            # vLLM reads the standard HuggingFace env variable.
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        print(f"[DeepSeekDeployment] Loading {MODEL_ID} "
              f"(tp={TENSOR_PARALLEL_SIZE}, dtype=bfloat16)...")
        try:
            engine_args = AsyncEngineArgs(
                model=MODEL_ID,
                dtype="bfloat16",
                trust_remote_code=True,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                # Disables CUDA graph capture for faster first startup. Remove
                # (or set to False) for maximum throughput once stable.
                enforce_eager=True,
                # The model's native max_seq_len is 163840, but only ~18944 KV
                # cache slots are available after loading weights on 2× A10G.
                # Cap to 16384 so vLLM's startup check passes; this is ample
                # for all practical prompts and chat conversations.
                max_model_len=16384,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        except Exception as exc:
            print(f"[DeepSeekDeployment] FATAL — engine failed to start: {exc!r}")
            raise

        # Tokenizer is fetched lazily on the first request (async call).
        self._tokenizer = None
        print("[DeepSeekDeployment] AsyncLLMEngine ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_tokenizer(self):
        """Fetch the tokenizer from the engine (cached after first call)."""
        if self._tokenizer is None:
            self._tokenizer = await self._engine.get_tokenizer()
        return self._tokenizer

    async def _build_prompt(self, messages: List[_Message]) -> str:
        """
        Apply the model's own chat template via the HuggingFace tokenizer.

        DeepSeek-V2 ships a Jinja2 chat template inside its tokenizer config;
        using it here ensures the exact same formatting as the official demo.
        """
        tokenizer = await self._get_tokenizer()
        raw = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(
            raw,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # OpenAI-compatible endpoints
    # ------------------------------------------------------------------

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(self, body: ChatCompletionRequest) -> JSONResponse:
        from vllm.sampling_params import SamplingParams

        prompt = await self._build_prompt(body.messages)
        sampling_params = SamplingParams(
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop or [],
            n=body.n,
        )
        request_id = str(uuid.uuid4())

        if body.stream:
            return StreamingResponse(
                self._stream_chunks(prompt, sampling_params, request_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming: consume the async generator and return the final output.
        final_text = ""
        async for output in self._engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                final_text = output.outputs[0].text

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "model": SERVED_MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })

    async def _stream_chunks(
        self,
        prompt: str,
        sampling_params,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """Yield SSE chunks as vLLM produces tokens."""
        prev_text = ""
        async for output in self._engine.generate(prompt, sampling_params, request_id):
            if not output.outputs:
                continue
            current_text = output.outputs[0].text
            delta = current_text[len(prev_text):]
            prev_text = current_text
            if not delta:
                continue
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": SERVED_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    @fastapi_app.get("/v1/models")
    async def list_models(self) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "owned_by": "anyscale",
            }],
        })

    @fastapi_app.get("/health")
    async def health(self) -> JSONResponse:
        return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Bind — this is what anyscale_service.yaml references as the import path
# ---------------------------------------------------------------------------

deployment = DeepSeekDeployment.bind()