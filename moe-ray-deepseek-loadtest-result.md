████████████████████████████████████████████████████████████████████████████████
 LOAD TEST SUITE
 Target: http://localhost:8080
 Model: deepseek
 Phases: [0, 1, 2, 3]
 Started: 2026-04-11T04:40:05.919512+00:00
████████████████████████████████████████████████████████████████████████████████

================================================================================
 PHASE 0: WARMUP
 Warming up containers, KV cache, and JIT compilation...
================================================================================
  Warmup request 1/5... ✓ (TTFT: 2766ms)
  Warmup request 2/5... ✓ (TTFT: 2786ms)
  Warmup request 3/5... ✓ (TTFT: 2595ms)
  Warmup request 4/5... ✓ (TTFT: 2597ms)
  Warmup request 5/5... ✓ (TTFT: 2651ms)

  Warmup complete: 5/5 successful
  (These results will be discarded)

================================================================================
 PHASE 1: SINGLE REQUEST BASELINE
 Sequential requests with varying prompt lengths (no concurrency)
================================================================================

  Testing prompt length ~32 tokens...
    Request 1: ✓ TTFT=4964ms Total=5518ms
    Request 2: ✓ TTFT=4937ms Total=5495ms
    Request 3: ✓ TTFT=4971ms Total=5531ms

  Testing prompt length ~64 tokens...
    Request 1: ✓ TTFT=4937ms Total=5501ms
    Request 2: ✓ TTFT=4964ms Total=5527ms
    Request 3: ✓ TTFT=4934ms Total=5497ms

  Testing prompt length ~128 tokens...
    Request 1: ✓ TTFT=4938ms Total=5517ms
    Request 2: ✓ TTFT=4969ms Total=5527ms
    Request 3: ✓ TTFT=4918ms Total=5478ms

  Testing prompt length ~256 tokens...
    Request 1: ✓ TTFT=5010ms Total=5569ms
    Request 2: ✓ TTFT=5229ms Total=5779ms
    Request 3: ✓ TTFT=5047ms Total=5608ms

  Testing prompt length ~512 tokens...
    Request 1: ✓ TTFT=5231ms Total=5783ms
    Request 2: ✓ TTFT=4992ms Total=5546ms
    Request 3: ✓ TTFT=5025ms Total=5613ms

════════════════════════════════════════════════════════════════════════════════
 Phase 1 Results: Baseline Latency by Prompt Length
════════════════════════════════════════════════════════════════════════════════
│ Prompt Tokens │ Count │ TTFT p50 │ TTFT p90 │ TTFT p99 │ Total p90 │ Total p99 │
├───────────────┼───────┼──────────┼──────────┼──────────┼───────────┼───────────┤
│       32      │   3   │  4964ms  │  4970ms  │  4971ms  │   5529ms  │   5531ms  │
│       64      │   3   │  4937ms  │  4958ms  │  4963ms  │   5522ms  │   5527ms  │
│      128      │   3   │  4938ms  │  4962ms  │  4968ms  │   5525ms  │   5527ms  │
│      256      │   3   │  5047ms  │  5193ms  │  5225ms  │   5744ms  │   5775ms  │
│      512      │   3   │  5025ms  │  5190ms  │  5227ms  │   5749ms  │   5780ms  │
└───────────────┴───────┴──────────┴──────────┴──────────┴───────────┴───────────┘

================================================================================
 PHASE 2: CONCURRENT WAVE TESTING
 Firing parallel requests to measure degradation under load
================================================================================

  Testing concurrency=1 (20 requests)...
    Success: 20/20 | TTFT p90: 5208ms | Total p90: 5755ms | Throughput: 6.6 tok/s

  Testing concurrency=2 (20 requests)...
    Success: 20/20 | TTFT p90: 5244ms | Total p90: 5802ms | Throughput: 6.4 tok/s

  Testing concurrency=4 (20 requests)...
    Success: 20/20 | TTFT p90: 5390ms | Total p90: 5944ms | Throughput: 6.3 tok/s

  Testing concurrency=8 (20 requests)...
    Success: 20/20 | TTFT p90: 5699ms | Total p90: 6285ms | Throughput: 6.0 tok/s

  Testing concurrency=16 (20 requests)...
    Success: 20/20 | TTFT p90: 11422ms | Total p90: 12004ms | Throughput: 4.2 tok/s

════════════════════════════════════════════════════════════════════════════════
 Phase 2 Results: Latency by Concurrency Level
════════════════════════════════════════════════════════════════════════════════
│ Concurrency │ Success% │ TTFT p50 │ TTFT p90 │ TTFT p99 │ Total p90 │ Total p99 │ Tok/s │
├─────────────┼──────────┼──────────┼──────────┼──────────┼───────────┼───────────┼───────┤
│      1      │   100%   │  5028ms  │  5208ms  │  5342ms  │   5755ms  │   5896ms  │  6.6  │
│      2      │   100%   │  5191ms  │  5244ms  │  5260ms  │   5802ms  │   5817ms  │  6.4  │
│      4      │   100%   │  5353ms  │  5390ms  │  5416ms  │   5944ms  │   5976ms  │  6.3  │
│      8      │   100%   │  5676ms  │  5699ms  │  5701ms  │   6285ms  │   6286ms  │  6.0  │
│      16     │   100%   │ 10234ms  │ 11422ms  │ 11437ms  │  12004ms  │  12017ms  │  4.2  │
└─────────────┴──────────┴──────────┴──────────┴──────────┴───────────┴───────────┴───────┘

================================================================================
 PHASE 3: SUSTAINED RPS TESTING
 Maintaining fixed request rate to find saturation point
================================================================================

  Testing 1.0 RPS for 30.0s...
    Sent: 30 | Actual RPS: 1.0 | Success: 30 | TTFT p90: 5648ms

  Testing 2.0 RPS for 30.0s...
    Sent: 60 | Actual RPS: 2.0 | Success: 60 | TTFT p90: 17643ms

  Testing 5.0 RPS for 30.0s...
    Sent: 150 | Actual RPS: 5.0 | Success: 117 | TTFT p90: 49210ms

  Testing 10.0 RPS for 30.0s...
    Sent: 300 | Actual RPS: 10.0 | Success: 100 | TTFT p90: 56056ms

════════════════════════════════════════════════════════════════════════════════
 Phase 3 Results: Latency by Sustained RPS
════════════════════════════════════════════════════════════════════════════════
│ Target RPS │ Actual RPS │ Success% │ TTFT p50 │ TTFT p90 │ TTFT p99 │ Total p90 │ Total p99 │
├────────────┼────────────┼──────────┼──────────┼──────────┼──────────┼───────────┼───────────┤
│    1.0     │    1.0     │   100%   │  5556ms  │  5648ms  │  5799ms  │   6210ms  │   6426ms  │
│    2.0     │    2.0     │   100%   │ 10953ms  │ 17643ms  │ 19658ms  │  18209ms  │  20222ms  │
│    5.0     │    5.0     │   78%    │ 27014ms  │ 49210ms  │ 54889ms  │  49784ms  │  55432ms  │
│    10.0    │    10.0    │   33%    │ 28815ms  │ 56056ms  │ 61271ms  │  56617ms  │  61828ms  │
└────────────┴────────────┴──────────┴──────────┴──────────┴──────────┴───────────┴───────────┘

████████████████████████████████████████████████████████████████████████████████
 FINAL SUMMARY
████████████████████████████████████████████████████████████████████████████████

  Total Requests: 660
  Total Success: 427 (64.7%)

  Combined Stats (excluding warmup):
    TTFT:  p50=11436ms  p90=48070ms  p99=57750ms
    Total: p50=12009ms  p90=48628ms  p99=58309ms
    TPOT:  p50=15.7ms  p90=16.3ms  p99=18.6ms
    Throughput: 3.5 tokens/sec (mean)

  Completed: 2026-04-11T04:49:40.500341+00:00
████████████████████████████████████████████████████████████████████████████████