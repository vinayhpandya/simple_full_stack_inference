# Load Test Suite Results

| Field | Value |
|-------|-------|
| **Target** | `http://localhost:80` |
| **Model** | `modal_vllm` |
| **Phases** | 0, 1, 2, 3 |
| **Started** | 2026-04-03T21:31:28 UTC |
| **Completed** | 2026-04-03T22:02:27 UTC |

---

## Phase 0: Warmup

Warming up containers, KV cache, and JIT compilation.

| Request | TTFT |
|---------|------|
| 1/5 | 7923ms |
| 2/5 | 7098ms |
| 3/5 | 11866ms |
| 4/5 | 3665ms |
| 5/5 | 9201ms |

✅ **Warmup complete: 5/5 successful** *(results discarded)*

---

## Phase 1: Single Request Baseline

Sequential requests with varying prompt lengths (no concurrency).

| Prompt Tokens | Count | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |
|:---:|:---:|---:|---:|---:|---:|---:|
| 32 | 3 | 6032ms | 34655ms | 41095ms | 35222ms | 41663ms |
| 64 | 3 | 11261ms | 11484ms | 11534ms | 12050ms | 12101ms |
| 128 | 3 | 8332ms | 10384ms | 10846ms | 10949ms | 11411ms |
| 256 | 3 | 10557ms | 10592ms | 10600ms | 11156ms | 11163ms |
| 512 | 3 | 9402ms | 9474ms | 9490ms | 10040ms | 10057ms |

> **Note:** The 32-token p90/p99 is heavily skewed by the first request (41810ms), likely a cold-start artifact.

---

## Phase 2: Concurrent Wave Testing

Firing parallel requests to measure degradation under load (20 requests per level).

| Concurrency | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 | Tok/s |
|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 100% | 8635ms | 11344ms | 12299ms | 11910ms | 12862ms | 3.9 |
| 2 | 100% | 15012ms | 17164ms | 19490ms | 17729ms | 20056ms | 2.4 |
| 4 | 100% | 23891ms | 60368ms | 60940ms | 60935ms | 61506ms | 1.4 |
| 8 | 100% | 63470ms | 93045ms | 96811ms | 93610ms | 97372ms | 0.7 |
| 16 | 75% | 56039ms | 103476ms | 114903ms | 104038ms | 115464ms | 1.1 |

> ⚠️ **Failures begin at concurrency=16** (5/20 requests failed). TTFT degrades sharply beyond concurrency=2.

---

## Phase 3: Sustained RPS Testing

Maintaining a fixed request rate over 30 seconds to find the saturation point.

| Target RPS | Actual RPS | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| 1.0 | 1.0 | 63% | 57136ms | 108314ms | 116259ms | 108875ms | 116821ms |
| 2.0 | 2.0 | 13% | 81812ms | 104012ms | 109742ms | 104575ms | 110304ms |
| 5.0 | 5.0 | 17% | 100714ms | 196171ms | 208146ms | 196732ms | 208708ms |
| 10.0 | 10.0 | 9% | 113463ms | 207147ms | 221058ms | 207708ms | 221622ms |

> ⚠️ **System is already saturated at 1.0 RPS** — only 63% success rate sustained. All higher rates show severe degradation.

---

## Final Summary

| Metric | Value |
|--------|-------|
| **Total Requests** | 660 |
| **Total Successes** | 194 |
| **Overall Success Rate** | 29.4% |

### Combined Stats *(excluding warmup)*

| Metric | p50 | p90 | p99 |
|--------|----:|----:|----:|
| **TTFT** | 34645ms | 153962ms | 211333ms |
| **Total** | 35214ms | 154524ms | 211895ms |
| **TPOT** | 18.2ms | 18.3ms | 18.7ms |

**Mean Throughput:** 1.5 tokens/sec