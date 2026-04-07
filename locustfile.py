"""
Model-agnostic Locust benchmark for the AI gateway.

All configuration is driven by environment variables so the sweep script
can control everything without touching this file.

Environment variables:
  MODEL_NAME      Model name sent in the request (must match gateway config).
                  Default: unset (gateway picks default).
  TASK_SET        One of: cold, warm, mixed. Default: mixed.
  MAX_TOKENS      Max tokens to generate per request. Default: 200.
  WAIT_MIN        Min wait between requests in seconds. Default: 1.
  WAIT_MAX        Max wait between requests in seconds. Default: 3.

Example:
  MODEL_NAME=tinyllama TASK_SET=warm MAX_TOKENS=200 locust -f locustfile.py \
    --host http://localhost:8080 --headless --users 10 --spawn-rate 2 --run-time 3m
"""

import os
import random
import uuid
from typing import Optional

from locust import HttpUser, TaskSet, between, events, task


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
MODEL_NAME: Optional[str] = os.environ.get("MODEL_NAME") or None
TASK_SET_NAME: str        = os.environ.get("TASK_SET", "mixed").lower()
MAX_TOKENS: int           = int(os.environ.get("MAX_TOKENS", "200"))
WAIT_MIN: float           = float(os.environ.get("WAIT_MIN", "1"))
WAIT_MAX: float           = float(os.environ.get("WAIT_MAX", "3"))

WARM_PROMPT_POOL = [
    "Explain the difference between TCP and UDP in simple terms.",
    "What are the main causes of the French Revolution?",
    "How does gradient descent work in machine learning?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "What is the difference between a mutex and a semaphore?",
]


# ---------------------------------------------------------------------------
# Custom metrics aggregator
# ---------------------------------------------------------------------------
class LLMMetrics:
    def __init__(self):
        self.ttft_samples: list[float]       = []
        self.tpot_samples: list[float]       = []
        self.queue_time_samples: list[float] = []

    def record(self, metrics: dict):
        if (v := metrics.get("ttft_ms")):       self.ttft_samples.append(v)
        if (v := metrics.get("tpot_ms")):       self.tpot_samples.append(v)
        if (v := metrics.get("queue_time_ms")): self.queue_time_samples.append(v)

    def _pct(self, samples: list[float]) -> dict:
        if not samples:
            return {"n": 0}
        s = sorted(samples)
        n = len(s)
        return {
            "n":    n,
            "mean": round(sum(s) / n, 2),
            "min":  round(s[0], 2),
            "p50":  round(s[int(n * 0.50)], 2),
            "p90":  round(s[int(n * 0.90)], 2),
            "p99":  round(s[int(n * 0.99)], 2),
            "max":  round(s[-1], 2),
        }

    def report(self) -> str:
        lines = [
            "\n========== LLM Custom Metrics ==========",
            f"TTFT (ms):       {self._pct(self.ttft_samples)}",
            f"TPOT (ms):       {self._pct(self.tpot_samples)}",
            f"Queue time (ms): {self._pct(self.queue_time_samples)}",
            "=========================================",
        ]
        return "\n".join(lines)


_metrics = LLMMetrics()


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    print(_metrics.report())


# ---------------------------------------------------------------------------
# Shared request helper
# ---------------------------------------------------------------------------
def chat_request(client, prompt: str, label: str) -> None:
    payload: dict = {
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream":     False,
    }
    if MODEL_NAME:
        payload["model"] = MODEL_NAME

    with client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Request-ID": str(uuid.uuid4())},
        name=label,
        catch_response=True,
    ) as resp:
        if resp.status_code != 200:
            resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
            return
        try:
            data = resp.json()
        except Exception as e:
            resp.failure(f"Invalid JSON: {e}")
            return

        llm_metrics = data.get("metrics", {})
        _metrics.record(llm_metrics)
        resp.success()


# ---------------------------------------------------------------------------
# Task sets
# ---------------------------------------------------------------------------
class ColdPrompts(TaskSet):
    """Every request gets a unique prompt — no LMCache hits possible."""

    @task
    def cold(self):
        prompt = (
            f"Write a short paragraph about this topic: {uuid.uuid4()}. "
            "Use at least three sentences."
        )
        chat_request(self.client, prompt, label=f"cold [{MODEL_NAME or 'default'}]")


class WarmPrompts(TaskSet):
    """Fixed prompt pool — LMCache should serve repeated prefill from CPU RAM."""

    @task
    def warm(self):
        prompt = random.choice(WARM_PROMPT_POOL)
        chat_request(self.client, prompt, label=f"warm [{MODEL_NAME or 'default'}]")


class MixedPrompts(TaskSet):
    """70 % warm / 30 % cold — approximates a realistic production workload."""

    @task(7)
    def warm(self):
        prompt = random.choice(WARM_PROMPT_POOL)
        chat_request(self.client, prompt, label=f"mixed-warm [{MODEL_NAME or 'default'}]")

    @task(3)
    def cold(self):
        prompt = f"Tell me something interesting about the number {random.randint(1, 99999)}."
        chat_request(self.client, prompt, label=f"mixed-cold [{MODEL_NAME or 'default'}]")


# ---------------------------------------------------------------------------
# Task set registry — maps TASK_SET env var to the right class
# ---------------------------------------------------------------------------
_TASK_SETS: dict[str, type[TaskSet]] = {
    "cold":  ColdPrompts,
    "warm":  WarmPrompts,
    "mixed": MixedPrompts,
}

_selected_task_set = _TASK_SETS.get(TASK_SET_NAME, MixedPrompts)
if TASK_SET_NAME not in _TASK_SETS:
    print(f"[locust] Unknown TASK_SET '{TASK_SET_NAME}', falling back to 'mixed'.")


# ---------------------------------------------------------------------------
# Single generic user class — all knobs set by env vars
# ---------------------------------------------------------------------------
class LLMUser(HttpUser):
    tasks     = [_selected_task_set]
    wait_time = between(WAIT_MIN, WAIT_MAX)
