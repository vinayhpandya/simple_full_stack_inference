#!/usr/bin/env bash
# sweep.sh — runs a Locust benchmark sweep across models and scenarios.
#
# Each row in SWEEP_MATRIX is one run:
#   model_name | task_set | users | spawn_rate | run_time | max_tokens | wait_min | wait_max | notes
#
# Results land in:
#   results/<timestamp>/<model>_<task_set>/
#     ├── locust_stats.csv
#     ├── locust_stats_history.csv
#     ├── locust_failures.csv
#     └── run_config.json     ← exact params used, for reproducibility
#
# Adding a new model (e.g. MoE): just append rows to SWEEP_MATRIX.
#
# Usage:
#   chmod +x sweep.sh
#   ./sweep.sh                        # run all entries in SWEEP_MATRIX
#   ./sweep.sh --dry-run              # print what would run without executing
#   GATEWAY_URL=http://x:8080 ./sweep.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Global config — override via env vars
# ---------------------------------------------------------------------------
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
LOCUST_FILE="${LOCUST_FILE:-locustfile.py}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
DRY_RUN=false

# Parse flags
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Sweep matrix
# Columns (tab or multi-space separated — parsed positionally below):
#   1  model_name   — matches your gateway config; "default" = no model field
#   2  task_set     — cold | warm | mixed
#   3  users        — peak concurrent users
#   4  spawn_rate   — users spawned per second
#   5  run_time     — locust --run-time value (e.g. 3m, 120s)
#   6  max_tokens   — max tokens per response
#   7  wait_min     — min think-time between requests (seconds)
#   8  wait_max     — max think-time between requests (seconds)
#   9  notes        — free-text label saved into run_config.json
#
# To add a MoE model later, append rows like:
#   "mixtral-8x7b"  "cold"  10  2  3m  200  1  3  "MoE cold baseline"
# ---------------------------------------------------------------------------
declare -a SWEEP_MATRIX=(
  # --- TinyLlama ---
  "modal_vllm      cold   5   1  3m  200  1  3  TinyLlama cold baseline"
  "modal_vllm      warm   5   1  3m  200  1  3  TinyLlama warm LMCache"
  "modal_vllm      mixed  5   1  3m  200  1  3  TinyLlama realistic mix"
  "modal_vllm      cold   10  2  3m  200  1  3  TinyLlama cold high load"
  "modal_vllm      warm   10  2  3m  200  1  3  TinyLlama warm high load"

  # --- Llama 3.1 8B ---
  "modal_vllm_lmcache  cold   5   1  3m  200  1  3  Llama31 cold baseline"
  "modal_vllm_lmcache  warm   5   1  3m  200  1  3  Llama31 warm LMCache"
  "modal_vllm_lmcache  mixed  5   1  3m  200  1  3  Llama31 realistic mix"
  "modal_vllm_lmcache  cold   10  2  3m  200  1  3  Llama31 cold high load"
  "modal_vllm_lmcache  warm   10  2  3m  200  1  3  Llama31 warm high load"

  # --- Add MoE rows here when ready, e.g.: ---
  # "mixtral-8x7b  cold   5  1  3m  200  1  3  MoE cold baseline"
  # "mixtral-8x7b  warm   5  1  3m  200  1  3  MoE warm LMCache"
  # "mixtral-8x7b  mixed  10  2  3m  200  1  3  MoE realistic mix"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log()  { echo "[sweep] $*"; }
info() { echo "[sweep] >>> $*"; }
hr()   { echo "[sweep] -------------------------------------------------------"; }

require_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "[sweep] ERROR: '$1' not found. Install with: pip install locust"
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
require_cmd uv
require_cmd python3

TOTAL=${#SWEEP_MATRIX[@]}
log "Gateway : $GATEWAY_URL"
log "Locust  : $LOCUST_FILE"
log "Runs    : $TOTAL"
log "Output  : $RESULTS_ROOT/$RUN_TIMESTAMP/"
$DRY_RUN && log "DRY RUN mode — nothing will execute."
hr

for i in "${!SWEEP_MATRIX[@]}"; do
  row="${SWEEP_MATRIX[$i]}"

  # Parse columns positionally (collapse runs of spaces)
  read -r model_name task_set users spawn_rate run_time max_tokens \
           wait_min wait_max notes <<< "$row"

  run_num=$((i + 1))
  # Sanitise model name for use as a directory component
  safe_model="${model_name//\//_}"
  out_dir="$RESULTS_ROOT/$RUN_TIMESTAMP/${safe_model}_${task_set}_u${users}"
  csv_prefix="$out_dir/locust"

  hr
  info "Run $run_num / $TOTAL"
  info "  Model      : $model_name"
  info "  Task set   : $task_set"
  info "  Users      : $users  Spawn rate: $spawn_rate/s  Duration: $run_time"
  info "  Max tokens : $max_tokens  Wait: ${wait_min}–${wait_max}s"
  info "  Notes      : $notes"
  info "  Output dir : $out_dir"

  if $DRY_RUN; then
    log "  [dry-run] skipping execution."
    continue
  fi

  mkdir -p "$out_dir"

  # Write run_config.json before the run so it exists even if locust crashes
  python3 - <<PYEOF
import json, sys
cfg = {
    "run_number":   $run_num,
    "timestamp":    "$RUN_TIMESTAMP",
    "gateway_url":  "$GATEWAY_URL",
    "model_name":   "$model_name",
    "task_set":     "$task_set",
    "users":        $users,
    "spawn_rate":   $spawn_rate,
    "run_time":     "$run_time",
    "max_tokens":   $max_tokens,
    "wait_min":     $wait_min,
    "wait_max":     $wait_max,
    "notes":        "$notes",
    "locust_file":  "$LOCUST_FILE",
}
with open("$out_dir/run_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(json.dumps(cfg, indent=2))
PYEOF

  # Run locust
  MODEL_NAME="$model_name"     \
  TASK_SET="$task_set"         \
  MAX_TOKENS="$max_tokens"     \
  WAIT_MIN="$wait_min"         \
  WAIT_MAX="$wait_max"         \
  uv run locust \
    -f "$LOCUST_FILE" \
    --host "$GATEWAY_URL" \
    --headless \
    --users "$users" \
    --spawn-rate "$spawn_rate" \
    --run-time "$run_time" \
    --csv "$csv_prefix" \
    --csv-full-history \
    2>&1 | tee "$out_dir/locust.log"

  log "Run $run_num complete → $out_dir"
done

hr
log "All $TOTAL runs complete."
log "Results: $RESULTS_ROOT/$RUN_TIMESTAMP/"

# Print a quick summary table from the stats CSVs
if ! $DRY_RUN; then
  log ""
  log "Quick summary (p50 / p95 latency ms, req/s, failures):"
  python3 - <<PYEOF
import csv, os, json, glob

root = "$RESULTS_ROOT/$RUN_TIMESTAMP"
rows = []

for stats_file in sorted(glob.glob(f"{root}/**/locust_stats.csv", recursive=True)):
    run_dir = os.path.dirname(stats_file)
    cfg_file = os.path.join(run_dir, "run_config.json")
    cfg = json.load(open(cfg_file))

    with open(stats_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated":
                rows.append({
                    "model":      cfg["model_name"],
                    "task_set":   cfg["task_set"],
                    "users":      cfg["users"],
                    "notes":      cfg["notes"],
                    "p50_ms":     row.get("50%", "-"),
                    "p95_ms":     row.get("95%", "-"),
                    "rps":        row.get("Requests/s", "-"),
                    "failures":   row.get("Failure Count", "-"),
                })

if not rows:
    print("  (no aggregated stats found)")
else:
    header = f"{'Model':<28} {'TaskSet':<8} {'Users':>5}  {'p50':>6}  {'p95':>6}  {'req/s':>7}  {'fails':>5}  Notes"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<28} {r['task_set']:<8} {r['users']:>5}"
            f"  {r['p50_ms']:>6}  {r['p95_ms']:>6}  {r['rps']:>7}  {r['failures']:>5}"
            f"  {r['notes']}"
        )
PYEOF
fi
