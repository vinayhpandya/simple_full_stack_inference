#!/usr/bin/env python3
"""
Load Testing Suite for simple_ai_gateway and simple_full_stack_inference

Phases:
  0 - Warmup: Discard first N requests to warm containers/caches
  1 - Single Request Baseline: Sequential requests with varying prompt lengths
  2 - Concurrent Wave Testing: Parallel requests to find degradation points
  3 - Sustained RPS: Fixed request rate for stability testing

Usage:
  python load_test.py --target http://localhost:8080 --model local
  python load_test.py --target http://localhost:8080 --model modal_vllm --phases 0,1,2,3
  python load_test.py --config load_test_config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WarmupConfig:
    """Phase 0: Warmup configuration"""
    num_requests: int = 5
    gap_ms: float = 500.0
    max_tokens: int = 64
    prompt_tokens: int = 32


@dataclass
class BaselineConfig:
    """Phase 1: Single request baseline configuration"""
    prompt_lengths: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    requests_per_length: int = 3
    max_tokens: int = 128


@dataclass
class WaveConfig:
    """Phase 2: Concurrent wave testing configuration"""
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    requests_per_wave: int = 20
    prompt_tokens: int = 64
    max_tokens: int = 128
    jitter_ms: tuple[int, int] = (0, 50)


@dataclass
class SustainedConfig:
    """Phase 3: Sustained RPS configuration"""
    rps_levels: list[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0])
    duration_per_level_s: float = 30.0
    prompt_tokens: int = 64
    max_tokens: int = 128


@dataclass
class LoadTestConfig:
    """Main configuration"""
    target_url: str = "http://localhost:8080"
    model: str = "local"
    timeout_s: float = 120.0
    phases: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    wave: WaveConfig = field(default_factory=WaveConfig)
    sustained: SustainedConfig = field(default_factory=SustainedConfig)


# ─────────────────────────────────────────────────────────────────────────────
# Token estimation and prompt generation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def generate_prompt(target_tokens: int, rng: random.Random) -> str:
    """Generate a prompt of approximately target_tokens length."""
    base_prompts = [
        "Explain the concept of {topic} in detail.",
        "Write a comprehensive analysis of {topic}.",
        "Describe the key aspects of {topic}.",
        "Provide an in-depth explanation of {topic}.",
        "Elaborate on the principles behind {topic}.",
    ]
    topics = [
        "machine learning algorithms",
        "distributed systems architecture",
        "database optimization techniques",
        "cloud computing paradigms",
        "software design patterns",
        "network protocols",
        "cryptographic systems",
        "concurrent programming",
        "data structures",
        "operating system concepts",
    ]

    prompt = rng.choice(base_prompts).format(topic=rng.choice(topics))

    # Pad with additional context to reach target tokens
    padding_phrases = [
        "Consider the historical context.",
        "Include practical examples.",
        "Discuss common misconceptions.",
        "Address performance implications.",
        "Cover edge cases and limitations.",
        "Explain the trade-offs involved.",
        "Describe best practices.",
        "Include relevant terminology.",
    ]

    while estimate_tokens(prompt) < target_tokens:
        prompt += " " + rng.choice(padding_phrases)

    # Trim if overshot
    while estimate_tokens(prompt) > target_tokens + 10:
        words = prompt.split()
        prompt = " ".join(words[:-5])

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# HTTP client and streaming request handler
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    """Result of a single request."""
    ok: bool
    error: Optional[str]
    ttft_s: float  # Time to first token
    total_s: float  # Total response time
    decode_s: float  # Time spent decoding (total - ttft)
    prompt_tokens_est: int
    output_tokens_est: int
    tpot_ms: float  # Time per output token (ms)
    tokens_per_s: float  # Output tokens per second
    phase: str
    metadata: dict = field(default_factory=dict)


async def streaming_chat_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> RequestResult:
    """Send a streaming chat completion request and measure latencies."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    parts: list[str] = []
    error: Optional[str] = None

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            timeout=httpx.Timeout(timeout),
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                error = f"HTTP {resp.status_code}: {body[:500].decode('utf-8', errors='replace')}"
            else:
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        error = str(chunk["error"])
                        break
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter()
                        parts.append(piece)
    except httpx.TimeoutException:
        error = "Request timed out"
    except httpx.ConnectError as e:
        error = f"Connection error: {e}"
    except Exception as e:
        error = str(e)

    t1 = time.perf_counter()
    text = "".join(parts)

    prompt_tokens = estimate_tokens(prompt)
    output_tokens = max(1, estimate_tokens(text))
    ttft_s = (ttft - t0) if ttft is not None else (t1 - t0)
    total_s = t1 - t0
    decode_s = max(0.0, total_s - ttft_s)

    # Time per output token (ms) - excluding first token
    decode_tokens = max(1, output_tokens - 1)
    tpot_ms = (decode_s / decode_tokens) * 1000 if decode_tokens > 0 else 0.0

    # Throughput
    tokens_per_s = output_tokens / total_s if total_s > 0 else 0.0

    return RequestResult(
        ok=error is None and bool(text),
        error=error,
        ttft_s=ttft_s,
        total_s=total_s,
        decode_s=decode_s,
        prompt_tokens_est=prompt_tokens,
        output_tokens_est=output_tokens,
        tpot_ms=tpot_ms,
        tokens_per_s=tokens_per_s,
        phase="",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def percentile(data: list[float], p: float) -> float:
    """Calculate the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def compute_stats(results: list[RequestResult]) -> dict[str, Any]:
    """Compute statistics from a list of results."""
    successful = [r for r in results if r.ok]
    if not successful:
        return {
            "count": len(results),
            "success_rate": 0.0,
            "ttft_p50": 0.0,
            "ttft_p90": 0.0,
            "ttft_p99": 0.0,
            "total_p50": 0.0,
            "total_p90": 0.0,
            "total_p99": 0.0,
            "tpot_p50": 0.0,
            "tpot_p90": 0.0,
            "tpot_p99": 0.0,
            "throughput_mean": 0.0,
        }

    ttfts = [r.ttft_s * 1000 for r in successful]  # Convert to ms
    totals = [r.total_s * 1000 for r in successful]
    tpots = [r.tpot_ms for r in successful]
    throughputs = [r.tokens_per_s for r in successful]

    return {
        "count": len(results),
        "success_count": len(successful),
        "success_rate": len(successful) / len(results) * 100,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_p99": percentile(ttfts, 99),
        "ttft_mean": statistics.mean(ttfts) if ttfts else 0.0,
        "total_p50": percentile(totals, 50),
        "total_p90": percentile(totals, 90),
        "total_p99": percentile(totals, 99),
        "total_mean": statistics.mean(totals) if totals else 0.0,
        "tpot_p50": percentile(tpots, 50),
        "tpot_p90": percentile(tpots, 90),
        "tpot_p99": percentile(tpots, 99),
        "tpot_mean": statistics.mean(tpots) if tpots else 0.0,
        "throughput_mean": statistics.mean(throughputs) if throughputs else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Print a formatted table."""
    if title:
        print(f"\n{'═' * 80}")
        print(f" {title}")
        print(f"{'═' * 80}")

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " │ ".join(h.center(widths[i]) for i, h in enumerate(headers))
    print(f"│ {header_line} │")
    print("├" + "┼".join("─" * (w + 2) for w in widths) + "┤")

    # Print rows
    for row in rows:
        row_line = " │ ".join(str(cell).center(widths[i]) for i, cell in enumerate(row))
        print(f"│ {row_line} │")

    print("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")


def format_ms(val: float) -> str:
    """Format milliseconds value."""
    if val < 1:
        return f"{val:.3f}"
    elif val < 100:
        return f"{val:.1f}"
    else:
        return f"{val:.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase implementations
# ─────────────────────────────────────────────────────────────────────────────

async def run_warmup(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    rng: random.Random,
) -> list[RequestResult]:
    """Phase 0: Warmup - sequential requests to warm containers."""
    print("\n" + "=" * 80)
    print(" PHASE 0: WARMUP")
    print(" Warming up containers, KV cache, and JIT compilation...")
    print("=" * 80)

    results: list[RequestResult] = []
    cfg = config.warmup

    for i in range(cfg.num_requests):
        print(f"  Warmup request {i + 1}/{cfg.num_requests}...", end=" ", flush=True)

        if cfg.gap_ms > 0 and i > 0:
            await asyncio.sleep(cfg.gap_ms / 1000.0)

        prompt = generate_prompt(cfg.prompt_tokens, rng)
        result = await streaming_chat_completion(
            client, config.target_url, config.model,
            prompt, cfg.max_tokens, config.timeout_s
        )
        result.phase = "warmup"
        results.append(result)

        status = "✓" if result.ok else f"✗ ({result.error})"
        print(f"{status} (TTFT: {result.ttft_s * 1000:.0f}ms)")

    success_count = sum(1 for r in results if r.ok)
    print(f"\n  Warmup complete: {success_count}/{cfg.num_requests} successful")
    print("  (These results will be discarded)")

    return results


async def run_baseline(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    rng: random.Random,
) -> list[RequestResult]:
    """Phase 1: Single request baseline - sequential requests with varying prompt lengths."""
    print("\n" + "=" * 80)
    print(" PHASE 1: SINGLE REQUEST BASELINE")
    print(" Sequential requests with varying prompt lengths (no concurrency)")
    print("=" * 80)

    results: list[RequestResult] = []
    cfg = config.baseline

    for prompt_len in cfg.prompt_lengths:
        print(f"\n  Testing prompt length ~{prompt_len} tokens...")
        phase_results: list[RequestResult] = []

        for i in range(cfg.requests_per_length):
            prompt = generate_prompt(prompt_len, rng)
            result = await streaming_chat_completion(
                client, config.target_url, config.model,
                prompt, cfg.max_tokens, config.timeout_s
            )
            result.phase = "baseline"
            result.metadata = {"prompt_target": prompt_len, "iteration": i}
            phase_results.append(result)
            results.append(result)

            status = "✓" if result.ok else "✗"
            print(f"    Request {i + 1}: {status} TTFT={result.ttft_s * 1000:.0f}ms Total={result.total_s * 1000:.0f}ms")

    # Print summary table
    headers = ["Prompt Tokens", "Count", "TTFT p50", "TTFT p90", "TTFT p99", "Total p90", "Total p99"]
    rows = []

    for prompt_len in cfg.prompt_lengths:
        phase_results = [r for r in results if r.metadata.get("prompt_target") == prompt_len]
        stats = compute_stats(phase_results)
        rows.append([
            str(prompt_len),
            str(stats["count"]),
            format_ms(stats["ttft_p50"]) + "ms",
            format_ms(stats["ttft_p90"]) + "ms",
            format_ms(stats["ttft_p99"]) + "ms",
            format_ms(stats["total_p90"]) + "ms",
            format_ms(stats["total_p99"]) + "ms",
        ])

    print_table(headers, rows, "Phase 1 Results: Baseline Latency by Prompt Length")

    return results


async def run_waves(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    rng: random.Random,
) -> list[RequestResult]:
    """Phase 2: Concurrent wave testing - parallel requests to find degradation."""
    print("\n" + "=" * 80)
    print(" PHASE 2: CONCURRENT WAVE TESTING")
    print(" Firing parallel requests to measure degradation under load")
    print("=" * 80)

    all_results: list[RequestResult] = []
    cfg = config.wave

    for concurrency in cfg.concurrency_levels:
        print(f"\n  Testing concurrency={concurrency} ({cfg.requests_per_wave} requests)...")

        sem = asyncio.Semaphore(concurrency)
        wave_results: list[RequestResult] = []

        async def make_request(idx: int) -> RequestResult:
            async with sem:
                # Add jitter
                jitter = rng.randint(cfg.jitter_ms[0], cfg.jitter_ms[1]) / 1000.0
                await asyncio.sleep(jitter)

                prompt = generate_prompt(cfg.prompt_tokens, rng)
                result = await streaming_chat_completion(
                    client, config.target_url, config.model,
                    prompt, cfg.max_tokens, config.timeout_s
                )
                result.phase = "wave"
                result.metadata = {"concurrency": concurrency, "request_idx": idx}
                return result

        # Fire all requests
        tasks = [make_request(i) for i in range(cfg.requests_per_wave)]
        wave_results = await asyncio.gather(*tasks)

        all_results.extend(wave_results)

        stats = compute_stats(wave_results)
        success = sum(1 for r in wave_results if r.ok)
        print(f"    Success: {success}/{cfg.requests_per_wave} | "
              f"TTFT p90: {stats['ttft_p90']:.0f}ms | "
              f"Total p90: {stats['total_p90']:.0f}ms | "
              f"Throughput: {stats['throughput_mean']:.1f} tok/s")

    # Print summary table
    headers = ["Concurrency", "Success%", "TTFT p50", "TTFT p90", "TTFT p99", "Total p90", "Total p99", "Tok/s"]
    rows = []

    for concurrency in cfg.concurrency_levels:
        wave_results = [r for r in all_results if r.metadata.get("concurrency") == concurrency]
        stats = compute_stats(wave_results)
        rows.append([
            str(concurrency),
            f"{stats['success_rate']:.0f}%",
            format_ms(stats["ttft_p50"]) + "ms",
            format_ms(stats["ttft_p90"]) + "ms",
            format_ms(stats["ttft_p99"]) + "ms",
            format_ms(stats["total_p90"]) + "ms",
            format_ms(stats["total_p99"]) + "ms",
            f"{stats['throughput_mean']:.1f}",
        ])

    print_table(headers, rows, "Phase 2 Results: Latency by Concurrency Level")

    return all_results


async def run_sustained_rps(
    client: httpx.AsyncClient,
    config: LoadTestConfig,
    rng: random.Random,
) -> list[RequestResult]:
    """Phase 3: Sustained RPS testing - fixed request rate over time."""
    print("\n" + "=" * 80)
    print(" PHASE 3: SUSTAINED RPS TESTING")
    print(" Maintaining fixed request rate to find saturation point")
    print("=" * 80)

    all_results: list[RequestResult] = []
    cfg = config.sustained

    for rps in cfg.rps_levels:
        print(f"\n  Testing {rps} RPS for {cfg.duration_per_level_s}s...")

        interval = 1.0 / max(0.01, rps)
        results: list[RequestResult] = []
        pending_tasks: list[asyncio.Task] = []

        t_start = time.perf_counter()
        t_end = t_start + cfg.duration_per_level_s
        next_send = t_start
        seq = 0

        async def make_request(sequence: int) -> RequestResult:
            prompt = generate_prompt(cfg.prompt_tokens, rng)
            result = await streaming_chat_completion(
                client, config.target_url, config.model,
                prompt, cfg.max_tokens, config.timeout_s
            )
            result.phase = "sustained"
            result.metadata = {"rps": rps, "seq": sequence}
            return result

        while time.perf_counter() < t_end:
            now = time.perf_counter()

            # Send request at scheduled time
            if now >= next_send:
                task = asyncio.create_task(make_request(seq))
                pending_tasks.append(task)
                seq += 1
                next_send += interval

            # Small sleep to not busy-wait
            await asyncio.sleep(0.01)

        # Wait for all pending requests to complete
        if pending_tasks:
            completed = await asyncio.gather(*pending_tasks)
            results.extend(completed)

        all_results.extend(results)

        stats = compute_stats(results)
        actual_rps = len(results) / cfg.duration_per_level_s
        success = sum(1 for r in results if r.ok)
        print(f"    Sent: {len(results)} | Actual RPS: {actual_rps:.1f} | "
              f"Success: {success} | TTFT p90: {stats['ttft_p90']:.0f}ms")

    # Print summary table
    headers = ["Target RPS", "Actual RPS", "Success%", "TTFT p50", "TTFT p90", "TTFT p99", "Total p90", "Total p99"]
    rows = []

    for rps in cfg.rps_levels:
        rps_results = [r for r in all_results if r.metadata.get("rps") == rps]
        stats = compute_stats(rps_results)
        actual_rps = len(rps_results) / cfg.duration_per_level_s if rps_results else 0
        rows.append([
            f"{rps:.1f}",
            f"{actual_rps:.1f}",
            f"{stats['success_rate']:.0f}%",
            format_ms(stats["ttft_p50"]) + "ms",
            format_ms(stats["ttft_p90"]) + "ms",
            format_ms(stats["ttft_p99"]) + "ms",
            format_ms(stats["total_p90"]) + "ms",
            format_ms(stats["total_p99"]) + "ms",
        ])

    print_table(headers, rows, "Phase 3 Results: Latency by Sustained RPS")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_load_test(config: LoadTestConfig) -> dict[str, Any]:
    """Run the complete load test suite."""
    print("\n" + "█" * 80)
    print(" LOAD TEST SUITE")
    print(f" Target: {config.target_url}")
    print(f" Model: {config.model}")
    print(f" Phases: {config.phases}")
    print(f" Started: {datetime.now(timezone.utc).isoformat()}")
    print("█" * 80)

    rng = random.Random(42)  # Fixed seed for reproducibility

    all_results: dict[str, list[RequestResult]] = {}

    async with httpx.AsyncClient() as client:
        # Phase 0: Warmup
        if 0 in config.phases:
            warmup_results = await run_warmup(client, config, rng)
            all_results["warmup"] = warmup_results

        # Phase 1: Single request baseline
        if 1 in config.phases:
            baseline_results = await run_baseline(client, config, rng)
            all_results["baseline"] = baseline_results

        # Phase 2: Concurrent waves
        if 2 in config.phases:
            wave_results = await run_waves(client, config, rng)
            all_results["waves"] = wave_results

        # Phase 3: Sustained RPS
        if 3 in config.phases:
            sustained_results = await run_sustained_rps(client, config, rng)
            all_results["sustained"] = sustained_results

    # Print final summary
    print("\n" + "█" * 80)
    print(" FINAL SUMMARY")
    print("█" * 80)

    total_requests = sum(len(r) for r in all_results.values())
    total_success = sum(sum(1 for req in r if req.ok) for r in all_results.values())

    print(f"\n  Total Requests: {total_requests}")
    print(f"  Total Success: {total_success} ({total_success / total_requests * 100:.1f}%)")

    # Combined stats for non-warmup results
    combined = []
    for phase, results in all_results.items():
        if phase != "warmup":
            combined.extend(results)

    if combined:
        stats = compute_stats(combined)
        print(f"\n  Combined Stats (excluding warmup):")
        print(f"    TTFT:  p50={stats['ttft_p50']:.0f}ms  p90={stats['ttft_p90']:.0f}ms  p99={stats['ttft_p99']:.0f}ms")
        print(f"    Total: p50={stats['total_p50']:.0f}ms  p90={stats['total_p90']:.0f}ms  p99={stats['total_p99']:.0f}ms")
        print(f"    TPOT:  p50={stats['tpot_p50']:.1f}ms  p90={stats['tpot_p90']:.1f}ms  p99={stats['tpot_p99']:.1f}ms")
        print(f"    Throughput: {stats['throughput_mean']:.1f} tokens/sec (mean)")

    print(f"\n  Completed: {datetime.now(timezone.utc).isoformat()}")
    print("█" * 80 + "\n")

    return {
        "config": {
            "target_url": config.target_url,
            "model": config.model,
            "phases": config.phases,
        },
        "results": {
            phase: [
                {
                    "ok": r.ok,
                    "error": r.error,
                    "ttft_ms": r.ttft_s * 1000,
                    "total_ms": r.total_s * 1000,
                    "tpot_ms": r.tpot_ms,
                    "prompt_tokens": r.prompt_tokens_est,
                    "output_tokens": r.output_tokens_est,
                    "metadata": r.metadata,
                }
                for r in results
            ]
            for phase, results in all_results.items()
        },
        "summary": compute_stats(combined) if combined else {},
    }

def save_markdown_report(results: dict[str, Any], output_file: str) -> None:
    """Save load test results as a formatted markdown report."""
    config = results["config"]
    summary = results["summary"]
    phases = results["results"]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("# Load Test Report\n")
    lines.append(f"| | |")
    lines.append(f"|---|---|")
    lines.append(f"| **Target** | `{config['target_url']}` |")
    lines.append(f"| **Model** | `{config['model']}` |")
    lines.append(f"| **Phases Run** | {', '.join(str(p) for p in config['phases'])} |")
    lines.append(f"| **Generated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |")
    lines.append("")

    # ── Phase 1: Baseline ─────────────────────────────────────────────────────
    if "baseline" in phases:
        lines.append("## Phase 1: Single Request Baseline\n")
        lines.append("Sequential requests with no concurrency — establishes best-case latency.\n")
        lines.append("| Prompt Tokens | Count | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |")
        lines.append("|---|---|---|---|---|---|---|")

        baseline = phases["baseline"]
        prompt_lengths = sorted(set(r["metadata"]["prompt_target"] for r in baseline))
        for length in prompt_lengths:
            group = [r for r in baseline if r["metadata"].get("prompt_target") == length]
            ttfts = sorted(r["ttft_ms"] for r in group if r["ok"])
            totals = sorted(r["total_ms"] for r in group if r["ok"])
            lines.append(
                f"| {length} | {len(group)} "
                f"| {percentile(ttfts, 50):.1f}ms "
                f"| {percentile(ttfts, 90):.1f}ms "
                f"| {percentile(ttfts, 99):.1f}ms "
                f"| {percentile(totals, 90):.0f}ms "
                f"| {percentile(totals, 99):.0f}ms |"
            )
        lines.append("")

    # ── Phase 2: Concurrent Waves ─────────────────────────────────────────────
    if "waves" in phases:
        lines.append("## Phase 2: Concurrent Wave Testing\n")
        lines.append("Parallel requests at increasing concurrency levels — reveals degradation under load.\n")
        lines.append("| Concurrency | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 | Tok/s |")
        lines.append("|---|---|---|---|---|---|---|---|")

        waves = phases["waves"]
        concurrency_levels = sorted(set(r["metadata"]["concurrency"] for r in waves))
        for level in concurrency_levels:
            group = [r for r in waves if r["metadata"].get("concurrency") == level]
            successful = [r for r in group if r["ok"]]
            success_rate = len(successful) / len(group) * 100 if group else 0
            ttfts = sorted(r["ttft_ms"] for r in successful)
            totals = sorted(r["total_ms"] for r in successful)
            # Recalculate tokens/s from total_ms and output_tokens
            toks = [r["output_tokens"] / (r["total_ms"] / 1000) for r in successful if r["total_ms"] > 0]
            mean_toks = statistics.mean(toks) if toks else 0
            lines.append(
                f"| {level} | {success_rate:.0f}% "
                f"| {percentile(ttfts, 50):.1f}ms "
                f"| {percentile(ttfts, 90):.1f}ms "
                f"| {percentile(ttfts, 99):.1f}ms "
                f"| {percentile(totals, 90):.0f}ms "
                f"| {percentile(totals, 99):.0f}ms "
                f"| {mean_toks:.1f} |"
            )
        lines.append("")

    # ── Phase 3: Sustained RPS ────────────────────────────────────────────────
    if "sustained" in phases:
        lines.append("## Phase 3: Sustained RPS Testing\n")
        lines.append("Fixed request rate held over time — finds the saturation point.\n")
        lines.append("| Target RPS | Actual RPS | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |")
        lines.append("|---|---|---|---|---|---|---|---|")

        sustained = phases["sustained"]
        rps_levels = sorted(set(r["metadata"]["rps"] for r in sustained))
        for rps in rps_levels:
            group = [r for r in sustained if r["metadata"].get("rps") == rps]
            successful = [r for r in group if r["ok"]]
            success_rate = len(successful) / len(group) * 100 if group else 0
            # Actual RPS = requests sent / duration (infer duration from metadata or config)
            actual_rps = len(group) / 30.0  # uses default duration — could be passed in
            ttfts = sorted(r["ttft_ms"] for r in successful)
            totals = sorted(r["total_ms"] for r in successful)
            lines.append(
                f"| {rps:.1f} | {actual_rps:.1f} | {success_rate:.0f}% "
                f"| {percentile(ttfts, 50):.1f}ms "
                f"| {percentile(ttfts, 90):.1f}ms "
                f"| {percentile(ttfts, 99):.1f}ms "
                f"| {percentile(totals, 90):.0f}ms "
                f"| {percentile(totals, 99):.0f}ms |"
            )
        lines.append("")

    # ── Final Summary ─────────────────────────────────────────────────────────
    if summary:
        lines.append("## Summary\n")
        lines.append(f"| Metric | p50 | p90 | p99 |")
        lines.append(f"|---|---|---|---|")
        lines.append(f"| **TTFT** | {summary['ttft_p50']:.0f}ms | {summary['ttft_p90']:.0f}ms | {summary['ttft_p99']:.0f}ms |")
        lines.append(f"| **Total Latency** | {summary['total_p50']:.0f}ms | {summary['total_p90']:.0f}ms | {summary['total_p99']:.0f}ms |")
        lines.append(f"| **TPOT** | {summary['tpot_p50']:.1f}ms | {summary['tpot_p90']:.1f}ms | {summary['tpot_p99']:.1f}ms |")
        lines.append("")
        lines.append(f"**Total Requests:** {summary['count']}  ")
        lines.append(f"**Success Rate:** {summary['success_rate']:.1f}%  ")
        lines.append(f"**Mean Throughput:** {summary['throughput_mean']:.1f} tokens/sec  ")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

def parse_args() -> LoadTestConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load testing suite for AI inference services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test against local gateway
  python load_test.py --target http://localhost:8080 --model local

  # Test against Modal vLLM backend
  python load_test.py --target http://localhost:8080 --model modal_vllm

  # Run only specific phases
  python load_test.py --target http://localhost:8080 --phases 1,2

  # Custom sustained RPS test
  python load_test.py --target http://localhost:8080 --phases 0,3 --rps 1,2,5 --duration 60
        """,
    )

    parser.add_argument(
        "--target", "-t",
        default="http://localhost:8080",
        help="Target URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--model", "-m",
        default="local",
        help="Model/backend to use (default: local)"
    )
    parser.add_argument(
        "--phases", "-p",
        default="0,1,2,3",
        help="Comma-separated phases to run (default: 0,1,2,3)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)"
    )

    # Warmup options
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)"
    )

    # Baseline options
    parser.add_argument(
        "--prompt-lengths",
        default="32,64,128,256,512",
        help="Comma-separated prompt lengths for baseline (default: 32,64,128,256,512)"
    )

    # Wave options
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels for waves (default: 1,2,4,8,16)"
    )
    parser.add_argument(
        "--wave-requests",
        type=int,
        default=20,
        help="Requests per wave (default: 20)"
    )

    # Sustained options
    parser.add_argument(
        "--rps",
        default="1,2,5,10",
        help="Comma-separated RPS levels for sustained test (default: 1,2,5,10)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration per RPS level in seconds (default: 30)"
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    config = LoadTestConfig(
        target_url=args.target,
        model=args.model,
        timeout_s=args.timeout,
        phases=[int(p) for p in args.phases.split(",")],
        warmup=WarmupConfig(
            num_requests=args.warmup_requests,
        ),
        baseline=BaselineConfig(
            prompt_lengths=[int(p) for p in args.prompt_lengths.split(",")],
        ),
        wave=WaveConfig(
            concurrency_levels=[int(c) for c in args.concurrency.split(",")],
            requests_per_wave=args.wave_requests,
        ),
        sustained=SustainedConfig(
            rps_levels=[float(r) for r in args.rps.split(",")],
            duration_per_level_s=args.duration,
        ),
    )

    return config, args.output


def main() -> int:
    """Main entry point."""
    config, output_file = parse_args()

    try:
        results = asyncio.run(run_load_test(config))

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_file}")

        return 0
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
