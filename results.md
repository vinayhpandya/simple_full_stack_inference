# Benchmarking LLaMA 3.1-8B with LMCache on Modal

## Command

```bash
uv run python benchmark.py --model modal_vllm_lmcache \
    --dataset-name sharegpt \
    --concurrency 1 2 4 \
    --request-rate 1 2 4 8 \
    --metric-percentiles 50 90 99
```

---

## Setup

| Parameter | Value |
|---|---|
| Gateway | http://localhost:8080 |
| Dataset | sharegpt |
| Percentiles | 50, 90, 99 |
| Requests per level | 20 |
| Max tokens | 512 |
| Timeout | 120s |
| Concurrency levels | 1, 2, 4 |
| Request rates | 1.0, 2.0, 4.0, 8.0 req/s |

---

## Dataset

- Downloaded ShareGPT dataset → `~/.cache/llm_bench/sharegpt.json`
- **Source:** `https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered`
- Loaded **500 conversations** (input tokens: 100–2048)

---

## Concurrency Sweep

### Progress

| Concurrency | Result | TTFT p50 | Lat p99 | Tok/s |
|---|---|---|---|---|
| 1 | ✓ 20/20 ok | 1048ms | 36343ms | 25.8 |
| 2 | ✓ 20/20 ok | 9072ms | 70454ms | 26.0 |
| 4 | ✓ 20/20 ok | 36295ms | 118684ms | 26.2 |

### Summary

| Concurrency | OK | Errors | TTFT p50 | TTFT p90 | TTFT p99 | Lat p50 | Lat p90 | Lat p99 | TPOT p50 | TPOT p90 | TPOT p99 | ITL p50 | ITL p90 | ITL p99 | Rate (req/s) | Total Tok/s | In p50 | In p90 | In p99 | Out p50 | Out p90 | Out p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20/20 | 0 | 1048ms | 1200ms | 1200ms | 910ms | 2700ms | 36343ms | 36ms | 36ms | 40ms | 0.x | 1x | 13x | 0.x | 25.8 | 2xx | 887 | 9xx | 222 | 7xx | 974 |
| 2 | 20/20 | 0 | 9072ms | 22xxx ms | 63xxx ms | 16xxx ms | 69xxx ms | 70454ms | 36ms | 39ms | 40ms | 0.x | 1x | 13x | 0.x | 26.0 | 3xx | 902 | 1xxx | 252 | 5xx | 16xx |
| 4 | 20/20 | 0 | 36295ms | 11xxxx ms | 11xxxx ms | 58xxx ms | 11xxxx ms | 118684ms | 36ms | 39ms | 40ms | 0.x | 1x | 13x | 0.x | 26.2 | 1xxx | 10xx | 1xxx | 363 | 1xxx | 24xx |

> ✅ **No breaking point detected** — system handled all concurrency levels.

Results saved → `./results/modal_vllm_lmcache_sharegpt_concurrency_20260408_141225.csv`

---

## Request-Rate Sweep (Poisson Arrivals)

### Progress

| Rate (req/s) | Result | TTFT p50 | Lat p99 | Tok/s | Errors |
|---|---|---|---|---|---|
| 1.0 | ⚠ 8/20 ok | 61675ms | 138715ms | 26.1 | 12 (60%) |

> ⚠ **Error rate 60% ≥ 50% — stopping early.**

### Summary

| Rate (req/s) | OK | Errors | TTFT p50 | TTFT p90 | TTFT p99 | Lat p50 | Lat p90 | Lat p99 | TPOT p50 | TPOT p90 | TPOT p99 | ITL p50 | ITL p90 | ITL p99 | Rate (req/s) | Total Tok/s | In p50 | In p90 | In p99 | Out p50 | Out p90 | Out p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ⚠ 1.0 | 8/20 | 60% | 61675ms | 117xxxms | 11xxxms | 734xxxms | 13xxxms | 138715ms | 36ms | 37ms | 37ms | 0.7x | 11x | 130x | 0.x | 26.1 | 252 | 1344 | 13xx | 444 | 740 | 740 |

> ⚠ **Breaking point detected at request_rate = 1.0 req/s**

Results saved → `./results/modal_vllm_lmcache_sharegpt_request_rate_20260408_141447.csv`