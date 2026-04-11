# Load Test Suite

| Field | Value |
|-------|-------|
| **Target** | `http://localhost:8080` |
| **Model** | `qwen_vllm_model` |
| **Phases** | 0, 1, 2, 3 |
| **Started** | 2026-04-11T19:41:22.339604+00:00 |
| **Completed** | 2026-04-11T19:48:25.080693+00:00 |

---

## Phase 0: Warmup

Warming up containers, KV cache, and JIT compilation.

| Request | Result | TTFT |
|---------|--------|------|
| 1/5 | ✓ | 472ms |
| 2/5 | ✓ | 388ms |
| 3/5 | ✓ | 429ms |
| 4/5 | ✓ | 455ms |
| 5/5 | ✓ | 384ms |

**Warmup complete: 5/5 successful** *(results discarded)*

---

## Phase 1: Single Request Baseline

Sequential requests with varying prompt lengths (no concurrency).

### Raw Requests

| Prompt Length | Request | TTFT | Total |
|--------------|---------|------|-------|
| ~32 tokens | 1 | 517ms | 4776ms |
| ~32 tokens | 2 | 400ms | 4723ms |
| ~32 tokens | 3 | 519ms | 4815ms |
| ~64 tokens | 1 | 394ms | 4661ms |
| ~64 tokens | 2 | 515ms | 4820ms |
| ~64 tokens | 3 | 412ms | 4714ms |
| ~128 tokens | 1 | 392ms | 4760ms |
| ~128 tokens | 2 | 395ms | 4740ms |
| ~128 tokens | 3 | 443ms | 4755ms |
| ~256 tokens | 1 | 411ms | 4725ms |
| ~256 tokens | 2 | 467ms | 4776ms |
| ~256 tokens | 3 | 433ms | 4762ms |
| ~512 tokens | 1 | 487ms | 4828ms |
| ~512 tokens | 2 | 565ms | 4876ms |
| ~512 tokens | 3 | 480ms | 4800ms |

### Results: Baseline Latency by Prompt Length

| Prompt Tokens | Count | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |
|:-------------:|:-----:|:--------:|:--------:|:--------:|:---------:|:---------:|
| 32 | 3 | 517ms | 518ms | 519ms | 4807ms | 4814ms |
| 64 | 3 | 412ms | 494ms | 513ms | 4799ms | 4818ms |
| 128 | 3 | 395ms | 433ms | 442ms | 4759ms | 4760ms |
| 256 | 3 | 433ms | 460ms | 467ms | 4774ms | 4776ms |
| 512 | 3 | 487ms | 549ms | 563ms | 4866ms | 4875ms |

---

## Phase 2: Concurrent Wave Testing

Firing parallel requests to measure degradation under load.

### Raw Results

| Concurrency | Requests | Success | TTFT p90 | Total p90 | Throughput |
|:-----------:|:--------:|:-------:|:--------:|:---------:|:----------:|
| 1 | 20 | 20/20 | 446ms | 4798ms | 36.5 tok/s |
| 2 | 20 | 20/20 | 477ms | 4842ms | 36.7 tok/s |
| 4 | 20 | 20/20 | 510ms | 4926ms | 37.3 tok/s |
| 8 | 20 | 20/20 | 623ms | 5076ms | 36.6 tok/s |
| 16 | 20 | 20/20 | 635ms | 5281ms | 34.2 tok/s |

### Results: Latency by Concurrency Level

| Concurrency | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 | Tok/s |
|:-----------:|:--------:|:--------:|:--------:|:--------:|:---------:|:---------:|:-----:|
| 1 | 100% | 408ms | 446ms | 469ms | 4798ms | 4999ms | 36.5 |
| 2 | 100% | 426ms | 477ms | 525ms | 4842ms | 4873ms | 36.7 |
| 4 | 100% | 470ms | 510ms | 549ms | 4926ms | 4931ms | 37.3 |
| 8 | 100% | 473ms | 623ms | 625ms | 5076ms | 5085ms | 36.6 |
| 16 | 100% | 564ms | 635ms | 653ms | 5281ms | 5290ms | 34.2 |

---

## Phase 3: Sustained RPS Testing

Maintaining fixed request rate to find saturation point.

### Raw Results

| Target RPS | Duration | Sent | Actual RPS | Success | TTFT p90 |
|:----------:|:--------:|:----:|:----------:|:-------:|:--------:|
| 1.0 | 30s | 30 | 1.0 | 30 | 477ms |
| 2.0 | 30s | 60 | 2.0 | 60 | 512ms |
| 5.0 | 30s | 150 | 5.0 | 150 | 519ms |
| 10.0 | 30s | 300 | 10.0 | 300 | 4919ms |

### Results: Latency by Sustained RPS

| Target RPS | Actual RPS | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |
|:----------:|:----------:|:--------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| 1.0 | 1.0 | 100% | 438ms | 477ms | 597ms | 5021ms | 5139ms |
| 2.0 | 2.0 | 100% | 437ms | 512ms | 586ms | 5387ms | 5468ms |
| 5.0 | 5.0 | 100% | 441ms | 519ms | 1045ms | 6267ms | 6498ms |
| 10.0 | 10.0 | 100% | 3015ms | 4919ms | 5821ms | 10827ms | 11665ms |

---

## Final Summary

| Metric | Value |
|--------|-------|
| Total Requests | 660 |
| Total Success | 660 (100.0%) |

### Combined Stats *(excluding warmup)*

| Metric | p50 | p90 | p99 |
|--------|:---:|:---:|:---:|
| TTFT | 490ms | 4374ms | 5544ms |
| Total | 6091ms | 10409ms | 11568ms |
| TPOT | 32.0ms | 39.4ms | 44.8ms |

**Mean Throughput:** 26.8 tokens/sec