# Load Test Report

| | |
|---|---|
| **Target** | `http://localhost:8080` |
| **Model** | `deepseek` |
| **Phases Run** | 2, 3 |
| **Generated** | 2026-04-11 06:17:48 UTC |

---

## Phase 2: Concurrent Wave Testing

Parallel requests at increasing concurrency levels — reveals degradation under load.

| Concurrency | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 | Tok/s |
|---|---|---|---|---|---|---|---|
| 1 | 100% | 406ms | 418ms | 614ms | 2069ms | 2200ms | 85.1 |
| 2 | 100% | 410ms | 474ms | 917ms | 2249ms | 2708ms | 71.2 |
| 4 | 100% | 526ms | 526ms | 629ms | 2943ms | 2967ms | 61.1 |
| 8 | 100% | 486ms | 597ms | 604ms | 3840ms | 3850ms | 47.3 |
| 16 | 100% | 632ms | 740ms | 771ms | 4839ms | 4871ms | 41.4 |

---

## Phase 3: Sustained RPS Testing

Fixed request rate held over time — finds the saturation point.

| Target RPS | Actual RPS | Success% | TTFT p50 | TTFT p90 | TTFT p99 | Total p90 | Total p99 |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 100% | 413ms | 461ms | 476ms | 2238ms | 2247ms |
| 2.0 | 2.0 | 100% | 454ms | 498ms | 541ms | 3739ms | 3770ms |
| 5.0 | 5.0 | 100% | 506ms | 555ms | 599ms | 5919ms | 5970ms |
| 10.0 | 10.0 | 100% | 675ms | 970ms | 1369ms | 9245ms | 9616ms |

---

## Summary

| Metric | p50 | p90 | p99 |
|---|---|---|---|
| **TTFT** | 526ms | 811ms | 1318ms |
| **Total Latency** | 5896ms | 9057ms | 9472ms |
| **TPOT** | 33.1ms | 50.4ms | 88.4ms |

**Total Requests:** 640
**Success Rate:** 100.0%
**Mean Throughput:** 34.5 tokens/sec