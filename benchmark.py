"""
LLM Gateway Stress Benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Mirrors the options provided by `vllm bench serve` and works with any
OpenAI-compatible gateway.

Dataset modes:
  fixed     — single hardcoded prompt (fast sanity check)
  sharegpt  — real multi-turn conversations (~300MB, auto-downloaded + cached)
  random    — synthetic variable-length prompts built from realistic English
              sentences; no download required

Sweep modes (can be combined):
  --concurrency   — fixed parallel slots, swept across levels
  --request-rate  — Poisson arrival rate (req/s), swept across levels

Examples:
  # Quick sanity check
  python benchmark.py --model modal_vllm_lmcache

  # ShareGPT dataset, concurrency sweep
  python benchmark.py --model modal_vllm_lmcache --dataset-name sharegpt

  # Random prompts, request-rate sweep
  python benchmark.py --model modal_vllm_lmcache --dataset-name random \\
      --input-len 512 --output-len 256 --request-rate 1 2 4 8

  # Both sweeps, custom percentiles
  python benchmark.py --model modal_vllm_lmcache --dataset-name sharegpt \\
      --concurrency 1 4 8 16 --request-rate 1 2 4 8 \\
      --metric-percentiles 50 90 99
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()

# ── Constants ──────────────────────────────────────────────────────────────────
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_CACHE = Path.home() / ".cache" / "llm_bench" / "sharegpt.json"

FIXED_PROMPT = (
    "Explain the CAP theorem in distributed systems in detail, covering "
    "consistency, availability, and partition tolerance with real-world examples."
)

# Realistic English sentences used to build random prompts.
# Varied domains so tokenization behaves like real workloads.
SENTENCE_POOL = [
    "The transformer architecture introduced the concept of self-attention, allowing models to weigh the relevance of different tokens in a sequence.",
    "Distributed systems must handle network partitions gracefully, often requiring trade-offs between consistency and availability.",
    "In reinforcement learning, an agent interacts with an environment to maximize cumulative reward over time.",
    "Kubernetes orchestrates containerized workloads by abstracting underlying infrastructure into a unified control plane.",
    "The Fourier transform decomposes a signal into its constituent frequencies, enabling analysis in the frequency domain.",
    "Gradient descent iteratively adjusts model parameters in the direction that minimizes the loss function.",
    "A relational database enforces referential integrity through foreign key constraints between tables.",
    "Latency in a distributed system is affected by network round-trip time, serialization overhead, and queue depth.",
    "Retrieval-augmented generation combines a dense retriever with a generative model to produce grounded responses.",
    "The attention mechanism computes a weighted sum of values based on the similarity between queries and keys.",
    "Load balancers distribute incoming traffic across multiple backend instances to improve reliability and throughput.",
    "An embedding vector represents a token or document in a continuous high-dimensional semantic space.",
    "Cache eviction policies such as LRU and LFU determine which entries are removed when the cache reaches capacity.",
    "Fine-tuning a pretrained language model on domain-specific data can significantly improve downstream task performance.",
    "A service mesh manages inter-service communication by injecting sidecar proxies into each pod.",
    "The softmax function converts raw logits into a probability distribution over the vocabulary.",
    "Consensus algorithms like Raft ensure that distributed nodes agree on a single value despite failures.",
    "Prompt engineering involves carefully crafting input text to elicit desired behavior from a language model.",
    "Quantization reduces model size by representing weights in lower precision formats such as INT8 or INT4.",
    "Asynchronous I/O allows a process to initiate an operation and continue executing without waiting for completion.",
    "Speculative decoding improves inference throughput by drafting multiple tokens and verifying them in parallel.",
    "A vector database indexes high-dimensional embeddings to enable fast approximate nearest-neighbor search.",
    "The KV cache stores key and value tensors from previous tokens to avoid recomputation during autoregressive decoding.",
    "Horizontal scaling adds more instances of a service, while vertical scaling increases the resources of a single instance.",
    "Token streaming returns generated text to the client incrementally, reducing perceived latency for long outputs.",
    "Backpressure mechanisms prevent a fast producer from overwhelming a slow consumer in a data pipeline.",
    "Continuous batching dynamically groups incoming requests to maximize GPU utilization during inference.",
    "A language model's context window limits the number of tokens it can attend to in a single forward pass.",
    "Observability encompasses logging, metrics, and distributed tracing to understand system behavior in production.",
    "Multi-head attention applies several attention functions in parallel and concatenates the resulting representations.",
    "The PageRank algorithm ranks nodes in a graph by the number and quality of links pointing to them.",
    "Flash attention computes the attention matrix in tiles to reduce memory bandwidth and improve throughput.",
    "Rate limiting protects a service from being overwhelmed by restricting the number of requests per time window.",
    "A tokenizer converts raw text into a sequence of integer IDs that a language model can process.",
    "Tensor parallelism splits model weights across multiple GPUs so that large models fit in aggregate memory.",
]


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    success: bool
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    tpot_ms: float = 0.0
    token_times_ms: list[float] = field(default_factory=list)
    error: str = ""


@dataclass
class LevelSummary:
    sweep_type: str          # "concurrency" or "request_rate"
    sweep_value: float       # concurrency level or req/s value
    total_requests: int
    success_count: int
    error_count: int
    error_rate: float
    requests_per_sec: float
    throughput_tokens_per_sec: float
    # percentile dicts keyed by percentile int, e.g. {50: 123.4, 90: 456.7, 99: 789.0}
    ttft_pcts: dict[int, float]
    latency_pcts: dict[int, float]
    tpot_pcts: dict[int, float]
    itl_pcts: dict[int, float]
    # token lengths
    input_tokens_pcts: dict[int, float]
    output_tokens_pcts: dict[int, float]


# ── Percentile helper ──────────────────────────────────────────────────────────
def percentiles(data: list[float], pcts: list[int]) -> dict[int, float]:
    if not data:
        return {p: 0.0 for p in pcts}
    s = sorted(data)
    result = {}
    for p in pcts:
        idx = min(int(len(s) * p / 100), len(s) - 1)
        result[p] = s[idx]
    return result


# ── Dataset: download ──────────────────────────────────────────────────────────
def download_sharegpt(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[yellow]Downloading ShareGPT dataset → {dest}[/yellow]")
    console.print(f"[dim]{SHAREGPT_URL}[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading...", total=None)

        def reporthook(block, block_size, total_size):
            if total_size > 0:
                progress.update(task, total=total_size, completed=block * block_size)

        urllib.request.urlretrieve(SHAREGPT_URL, dest, reporthook)

    console.print("[green]✓ Download complete[/green]\n")


# ── Dataset: ShareGPT ──────────────────────────────────────────────────────────
def load_sharegpt(
    path: Path,
    min_input_tokens: int,
    max_input_tokens: int,
    num_prompts: int,
    seed: int,
) -> list[dict]:
    console.print(f"[dim]Loading ShareGPT dataset from {path}...[/dim]")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    random.seed(seed)
    random.shuffle(raw)

    def est_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

    pool: list[dict] = []
    for entry in raw:
        conversations = entry.get("conversations", [])
        if not conversations:
            continue
        messages = []
        for turn in conversations:
            role    = turn.get("from", "")
            content = turn.get("value", "")
            if not content:
                continue
            if role == "human":
                messages.append({"role": "user", "content": content})
            elif role == "gpt":
                messages.append({"role": "assistant", "content": content})
        if not messages:
            continue
        input_text = " ".join(m["content"] for m in messages if m["role"] == "user")
        est = est_tokens(input_text)
        if est < min_input_tokens or est > max_input_tokens:
            continue
        pool.append({"messages": messages, "estimated_input_tokens": est})
        if len(pool) >= num_prompts * 3:
            break

    if len(pool) < num_prompts:
        console.print(
            f"[yellow]⚠ Only {len(pool)} conversations matched filters "
            f"(wanted {num_prompts})[/yellow]"
        )
    else:
        pool = random.sample(pool, num_prompts)

    console.print(
        f"[green]✓ Loaded {len(pool)} ShareGPT conversations "
        f"(input tokens: {min_input_tokens}–{max_input_tokens})[/green]\n"
    )
    return pool


# ── Dataset: random realistic prompts ─────────────────────────────────────────
def build_random_pool(
    num_prompts: int,
    input_len: int,
    seed: int,
) -> list[dict]:
    """
    Build prompts by sampling and concatenating realistic sentences from
    SENTENCE_POOL until we reach approximately `input_len` tokens.
    Token length is estimated as words * 1.3.
    """
    random.seed(seed)
    pool: list[dict] = []
    for _ in range(num_prompts):
        words_target = int(input_len / 1.3)
        words = 0
        parts: list[str] = []
        while words < words_target:
            sentence = random.choice(SENTENCE_POOL)
            parts.append(sentence)
            words += len(sentence.split())
        text = " ".join(parts)
        est  = int(len(text.split()) * 1.3)
        pool.append({
            "messages": [{"role": "user", "content": text}],
            "estimated_input_tokens": est,
        })
    console.print(
        f"[green]✓ Built {num_prompts} random prompts "
        f"(target input tokens: ~{input_len})[/green]\n"
    )
    return pool


# ── Dataset: fixed ─────────────────────────────────────────────────────────────
def build_fixed_pool() -> list[dict]:
    return [{
        "messages": [{"role": "user", "content": FIXED_PROMPT}],
        "estimated_input_tokens": int(len(FIXED_PROMPT.split()) * 1.3),
    }]


# ── Core request ───────────────────────────────────────────────────────────────
async def send_request(
    client: httpx.AsyncClient,
    gateway_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    estimated_input_tokens: int,
    output_len: int,
    timeout: int,
) -> RequestResult:
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "max_tokens": output_len,
    }
    start = time.perf_counter()
    ttft_ms: Optional[float] = None
    completion_tokens = 0
    token_times: list[float] = []

    try:
        async with client.stream(
            "POST",
            f"{gateway_url}/v1/chat/completions",
            json=payload,
            headers={
                "Authorization":  f"Bearer {api_key}",
                "Content-Type":   "application/json",
            },
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        now = time.perf_counter()
                        if ttft_ms is None:
                            ttft_ms = (now - start) * 1000
                        token_times.append((now - start) * 1000)
                        completion_tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass

        total_ms = (time.perf_counter() - start) * 1000
        ttft = ttft_ms or total_ms
        decode_ms = total_ms - ttft
        tpot_ms = decode_ms / max(completion_tokens - 1, 1)

        # inter-token latency in ms between consecutive tokens
        itl_ms: list[float] = []
        if len(token_times) > 1:
            itl_ms = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]

        return RequestResult(
            success=True,
            ttft_ms=ttft,
            total_latency_ms=total_ms,
            completion_tokens=completion_tokens,
            prompt_tokens=estimated_input_tokens,
            tpot_ms=tpot_ms,
            token_times_ms=itl_ms,
        )

    except Exception as exc:
        total_ms = (time.perf_counter() - start) * 1000
        return RequestResult(success=False, total_latency_ms=total_ms, error=str(exc))


# ── Concurrency sweep: one level ───────────────────────────────────────────────
async def run_concurrency_level(
    cfg: "BenchConfig",
    concurrency: int,
    prompt_pool: list[dict],
) -> LevelSummary:
    console.print(
        f"  [cyan]concurrency={concurrency:>3}[/cyan]  "
        f"firing {cfg.requests_per_level} requests...",
        end=" ",
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(client, msgs, est):
        async with semaphore:
            return await send_request(
                client, cfg.gateway_url, cfg.api_key,
                cfg.model, msgs, est, cfg.output_len, cfg.timeout,
            )

    samples = random.choices(prompt_pool, k=cfg.requests_per_level)
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[bounded(client, s["messages"], s["estimated_input_tokens"]) for s in samples]
        )
        elapsed = time.perf_counter() - t0

    return _summarise("concurrency", concurrency, results, elapsed, cfg.metric_percentiles)


# ── Request-rate sweep: one level (Poisson arrivals) ──────────────────────────
async def run_request_rate_level(
    cfg: "BenchConfig",
    rate: float,
    prompt_pool: list[dict],
) -> LevelSummary:
    console.print(
        f"  [cyan]rate={rate:>6.1f} req/s[/cyan]  "
        f"firing {cfg.requests_per_level} requests...",
        end=" ",
    )
    samples  = random.choices(prompt_pool, k=cfg.requests_per_level)
    results  = []
    tasks    = []
    interval = 1.0 / rate  # mean inter-arrival time

    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()

        async def fire(msgs, est):
            return await send_request(
                client, cfg.gateway_url, cfg.api_key,
                cfg.model, msgs, est, cfg.output_len, cfg.timeout,
            )

        for s in samples:
            tasks.append(asyncio.create_task(fire(s["messages"], s["estimated_input_tokens"])))
            # Poisson: inter-arrival ~ Exponential(rate)
            await asyncio.sleep(random.expovariate(rate))

        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    return _summarise("request_rate", rate, results, elapsed, cfg.metric_percentiles)


# ── Summarise results for one level ───────────────────────────────────────────
def _summarise(
    sweep_type: str,
    sweep_value: float,
    results: tuple,
    elapsed: float,
    pcts: list[int],
) -> LevelSummary:
    successes = [r for r in results if r.success]
    errors    = [r for r in results if not r.success]
    n         = len(results)

    ttfts         = [r.ttft_ms for r in successes]
    latencies     = [r.total_latency_ms for r in successes]
    tpots         = [r.tpot_ms for r in successes]
    itls          = [v for r in successes for v in r.token_times_ms]
    input_toks    = [r.prompt_tokens for r in successes if r.prompt_tokens > 0]
    output_toks   = [r.completion_tokens for r in successes]
    total_tokens  = sum(output_toks)

    summary = LevelSummary(
        sweep_type=sweep_type,
        sweep_value=sweep_value,
        total_requests=n,
        success_count=len(successes),
        error_count=len(errors),
        error_rate=len(errors) / n * 100 if n else 0,
        requests_per_sec=n / elapsed if elapsed else 0,
        throughput_tokens_per_sec=total_tokens / elapsed if elapsed else 0,
        ttft_pcts=percentiles(ttfts, pcts),
        latency_pcts=percentiles(latencies, pcts),
        tpot_pcts=percentiles(tpots, pcts),
        itl_pcts=percentiles(itls, pcts),
        input_tokens_pcts=percentiles(input_toks, pcts),
        output_tokens_pcts=percentiles(output_toks, pcts),
    )

    # inline status
    p99_lat = summary.latency_pcts.get(99, 0)
    p50_ttft = summary.ttft_pcts.get(50, 0)
    status = (
        f"[green]✓[/green] {len(successes)}/{n} ok  "
        f"ttft_p50={p50_ttft:.0f}ms  "
        f"lat_p99={p99_lat:.0f}ms  "
        f"tok/s={summary.throughput_tokens_per_sec:.1f}"
    )
    if errors:
        status += f"  [red]{len(errors)} errors[/red]"
    console.print(status)

    return summary


# ── Breaking point detection ───────────────────────────────────────────────────
def detect_breaking_point(summaries: list[LevelSummary]) -> Optional[float]:
    for i, s in enumerate(summaries):
        if s.error_rate > 10:
            return s.sweep_value
        if i > 0:
            prev = summaries[i - 1]
            p99      = s.latency_pcts.get(99, 0)
            prev_p99 = prev.latency_pcts.get(99, 0)
            if prev_p99 > 0 and p99 > prev_p99 * 3:
                return s.sweep_value
    return None


# ── Rich table ─────────────────────────────────────────────────────────────────
def print_table(
    summaries: list[LevelSummary],
    breaking_point: Optional[float],
    pcts: list[int],
    dataset_mode: str,
    title: str,
):
    table = Table(
        title=title,
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )

    # first column label depends on sweep type
    sweep_label = "Conc." if summaries[0].sweep_type == "concurrency" else "Rate (r/s)"
    table.add_column(sweep_label, justify="right")
    table.add_column("OK/Total",  justify="right")
    table.add_column("Err%",      justify="right")

    for p in pcts:
        table.add_column(f"TTFT p{p}",  justify="right")
    for p in pcts:
        table.add_column(f"Lat p{p}",   justify="right")
    for p in pcts:
        table.add_column(f"TPOT p{p}",  justify="right")
    for p in pcts:
        table.add_column(f"ITL p{p}",   justify="right")

    table.add_column("Req/s",  justify="right")
    table.add_column("Tok/s",  justify="right")

    if dataset_mode != "fixed":
        for p in pcts:
            table.add_column(f"In p{p}",  justify="right")
        for p in pcts:
            table.add_column(f"Out p{p}", justify="right")

    for s in summaries:
        is_breaking = s.sweep_value == breaking_point
        val_str = (
            str(int(s.sweep_value)) if s.sweep_type == "concurrency"
            else f"{s.sweep_value:.1f}"
        )
        label = f"[bold red]⚠ {val_str}[/bold red]" if is_breaking else val_str

        row = [
            label,
            f"{s.success_count}/{s.total_requests}",
            f"{s.error_rate:.1f}%",
        ]
        for p in pcts:
            row.append(f"{s.ttft_pcts.get(p, 0):.0f}ms")
        for p in pcts:
            row.append(f"{s.latency_pcts.get(p, 0):.0f}ms")
        for p in pcts:
            row.append(f"{s.tpot_pcts.get(p, 0):.1f}ms")
        for p in pcts:
            row.append(f"{s.itl_pcts.get(p, 0):.1f}ms")

        row += [
            f"{s.requests_per_sec:.2f}",
            f"{s.throughput_tokens_per_sec:.1f}",
        ]

        if dataset_mode != "fixed":
            for p in pcts:
                row.append(f"{s.input_tokens_pcts.get(p, 0):.0f}")
            for p in pcts:
                row.append(f"{s.output_tokens_pcts.get(p, 0):.0f}")

        table.add_row(*row, style="bold red" if is_breaking else "")

    console.print()
    console.print(table)

    if breaking_point:
        console.print(
            f"\n[bold red]⚠  Breaking point detected at "
            f"{summaries[0].sweep_type}={breaking_point}[/bold red]"
        )
    else:
        console.print(
            "\n[bold green]✓  No breaking point detected — "
            "system handled all levels[/bold green]"
        )


# ── CSV ────────────────────────────────────────────────────────────────────────
def save_csv(
    summaries: list[LevelSummary],
    model: str,
    dataset_mode: str,
    sweep_type: str,
    pcts: list[int],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = model.replace("/", "_").replace(":", "_")
    path = f"{output_dir}/{safe}_{dataset_mode}_{sweep_type}_{ts}.csv"

    base_fields = [sweep_type, "total_requests", "success_count", "error_count", "error_rate_%"]
    for metric in ("ttft", "latency", "tpot", "itl"):
        for p in pcts:
            base_fields.append(f"{metric}_p{p}_ms")
    base_fields += ["requests_per_sec", "throughput_tokens_per_sec"]

    token_fields = []
    if dataset_mode != "fixed":
        for p in pcts:
            token_fields.append(f"input_tokens_p{p}")
        for p in pcts:
            token_fields.append(f"output_tokens_p{p}")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base_fields + token_fields)
        for s in summaries:
            row = [
                s.sweep_value, s.total_requests, s.success_count,
                s.error_count, round(s.error_rate, 2),
            ]
            for pct_dict in (s.ttft_pcts, s.latency_pcts, s.tpot_pcts, s.itl_pcts):
                for p in pcts:
                    row.append(round(pct_dict.get(p, 0), 2))
            row += [round(s.requests_per_sec, 3), round(s.throughput_tokens_per_sec, 2)]

            if dataset_mode != "fixed":
                for p in pcts:
                    row.append(round(s.input_tokens_pcts.get(p, 0)))
                for p in pcts:
                    row.append(round(s.output_tokens_pcts.get(p, 0)))

            w.writerow(row)

    console.print(f"[dim]Results saved → {path}[/dim]")
    return path


# ── Config container ───────────────────────────────────────────────────────────
@dataclass
class BenchConfig:
    model:              str
    gateway_url:        str
    api_key:            str
    dataset_mode:       str
    requests_per_level: int
    output_len:         int
    timeout:            int
    metric_percentiles: list[int]
    results_dir:        str


# ── Orchestrator ───────────────────────────────────────────────────────────────
async def run_benchmark(
    cfg: BenchConfig,
    prompt_pool: list[dict],
    concurrency_levels: list[int],
    request_rates: list[float],
):
    console.rule(f"[bold]🚀 Benchmarking [cyan]{cfg.model}[/cyan]")
    console.print(f"  Gateway     : [cyan]{cfg.gateway_url}[/cyan]")
    console.print(f"  Dataset     : [yellow]{cfg.dataset_mode}[/yellow]")
    console.print(f"  Percentiles : {cfg.metric_percentiles}")
    console.print(f"  Req/level   : {cfg.requests_per_level}")
    console.print(f"  Max tokens  : {cfg.output_len}")
    console.print(f"  Timeout     : {cfg.timeout}s")
    if concurrency_levels:
        console.print(f"  Concurrency : {concurrency_levels}")
    if request_rates:
        console.print(f"  Req rates   : {request_rates} req/s")
    console.print()

    # ── Concurrency sweep ──────────────────────────────────────────────────────
    if concurrency_levels:
        console.print("[bold underline]Concurrency Sweep[/bold underline]")
        c_summaries: list[LevelSummary] = []
        for c in concurrency_levels:
            summary = await run_concurrency_level(cfg, c, prompt_pool)
            c_summaries.append(summary)
            if summary.error_rate >= 50:
                console.print(
                    f"[red]  Error rate {summary.error_rate:.0f}% ≥ 50% — stopping early.[/red]"
                )
                break

        bp = detect_breaking_point(c_summaries)
        print_table(
            c_summaries, bp, cfg.metric_percentiles, cfg.dataset_mode,
            "📊 Concurrency Sweep Summary",
        )
        save_csv(c_summaries, cfg.model, cfg.dataset_mode, "concurrency",
                 cfg.metric_percentiles, cfg.results_dir)

    # ── Request-rate sweep ─────────────────────────────────────────────────────
    if request_rates:
        console.print("\n[bold underline]Request-Rate Sweep (Poisson arrivals)[/bold underline]")
        r_summaries: list[LevelSummary] = []
        for rate in request_rates:
            summary = await run_request_rate_level(cfg, rate, prompt_pool)
            r_summaries.append(summary)
            if summary.error_rate >= 50:
                console.print(
                    f"[red]  Error rate {summary.error_rate:.0f}% ≥ 50% — stopping early.[/red]"
                )
                break

        bp = detect_breaking_point(r_summaries)
        print_table(
            r_summaries, bp, cfg.metric_percentiles, cfg.dataset_mode,
            "📊 Request-Rate Sweep Summary",
        )
        save_csv(r_summaries, cfg.model, cfg.dataset_mode, "request_rate",
                 cfg.metric_percentiles, cfg.results_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM Gateway Stress Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick sanity check (fixed prompt, concurrency sweep)
  python benchmark.py --model modal_vllm_lmcache

  # ShareGPT dataset, concurrency + request-rate sweep
  python benchmark.py --model modal_vllm_lmcache --dataset-name sharegpt \\
      --concurrency 1 4 8 16 --request-rate 1 2 4 8

  # Random realistic prompts, custom input length
  python benchmark.py --model modal_vllm_lmcache --dataset-name random \\
      --input-len 512 --output-len 256

  # Custom percentiles
  python benchmark.py --model modal_vllm_lmcache --dataset-name sharegpt \\
      --metric-percentiles 50 90 99

  # ShareGPT with local dataset file (skip download)
  python benchmark.py --model modal_vllm_lmcache --dataset-name sharegpt \\
      --dataset-path /path/to/sharegpt.json
        """,
    )

    # Core
    parser.add_argument("--model",       required=True, help="Model name from your gateway config")
    parser.add_argument("--gateway-url", default="http://localhost:8080")
    parser.add_argument("--api-key",     default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--timeout",     type=int, default=120)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--seed",        type=int, default=42)

    # Dataset
    parser.add_argument(
        "--dataset-name",
        choices=["fixed", "sharegpt", "random"],
        default="fixed",
        help=(
            "fixed    — single hardcoded prompt (default, no download)\n"
            "sharegpt — real multi-turn conversations (~300MB, auto-downloaded + cached)\n"
            "random   — synthetic variable-length prompts, no download"
        ),
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=DEFAULT_CACHE,
        help=f"Path to local ShareGPT JSON (default: {DEFAULT_CACHE}). "
             "Only used with --dataset-name sharegpt.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=500,
        help="Number of prompts to sample from dataset (default: 500)",
    )
    parser.add_argument(
        "--input-len", type=int, default=256,
        help="Target input token length for random mode (default: 256)",
    )
    parser.add_argument(
        "--output-len", type=int, default=512,
        help="max_tokens sent to the model (default: 512)",
    )
    parser.add_argument(
        "--min-input-tokens", type=int, default=100,
        help="Min input tokens filter for sharegpt mode (default: 100)",
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=2048,
        help="Max input tokens filter for sharegpt mode (default: 2048)",
    )

    # Sweep modes
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
        help="Concurrency levels to sweep (default: 1 2 4 8 16 32). "
             "Pass 0 to disable concurrency sweep.",
    )
    parser.add_argument(
        "--request-rate", nargs="+", type=float, default=[],
        help="Request rates (req/s) to sweep with Poisson arrivals, e.g. --request-rate 1 2 4 8. "
             "Disabled by default.",
    )
    parser.add_argument(
        "--requests", type=int, default=20,
        help="Number of requests per concurrency/rate level (default: 20)",
    )

    # Metrics
    parser.add_argument(
        "--metric-percentiles", nargs="+", type=int, default=[50, 99],
        help="Percentiles to report for all metrics (default: 50 99)",
    )

    args = parser.parse_args()

    # Validate: at least one sweep mode must be active
    concurrency_levels = [c for c in args.concurrency if c > 0]
    request_rates      = args.request_rate

    if not concurrency_levels and not request_rates:
        parser.error("No sweep mode active. Provide --concurrency or --request-rate values.")

    # Build prompt pool
    dataset_mode = args.dataset_name
    if dataset_mode == "sharegpt":
        if not args.dataset_path.exists():
            download_sharegpt(args.dataset_path)
        prompt_pool = load_sharegpt(
            args.dataset_path,
            min_input_tokens=args.min_input_tokens,
            max_input_tokens=args.max_input_tokens,
            num_prompts=args.num_prompts,
            seed=args.seed,
        )
    elif dataset_mode == "random":
        prompt_pool = build_random_pool(
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            seed=args.seed,
        )
    else:
        prompt_pool = build_fixed_pool()

    cfg = BenchConfig(
        model=args.model,
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        dataset_mode=dataset_mode,
        requests_per_level=args.requests,
        output_len=args.output_len,
        timeout=args.timeout,
        metric_percentiles=sorted(args.metric_percentiles),
        results_dir=args.results_dir,
    )

    asyncio.run(run_benchmark(cfg, prompt_pool, concurrency_levels, request_rates))


if __name__ == "__main__":
    main()