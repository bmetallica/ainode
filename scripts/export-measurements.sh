#!/usr/bin/env bash
# Export everything this cluster has measured — to hand on, e.g. to improve
# the planner's estimates for other models.
#
# One JSON file: per model and node what it took (memory per node, the
# engine's own weights/cache/tokens figures, speed, load time, guard stops,
# the last loads) and how it was started (split, window, KV dtype, memory
# fraction, --max-num-seqs, every flag); the checkpoint's facts; what the
# planner's arithmetic predicts for exactly that launch; the nodes, the engine
# build, what runs now and the profiles. No credentials, keys or passwords.
#
# Run it on the head: requests from the node itself need no sign-in.
#
# Usage: scripts/export-measurements.sh [outfile] [host:port]
set -euo pipefail

OUT="${1:-ainode-measurements-$(date +%Y%m%d-%H%M).json}"
HOST="${2:-localhost:3000}"

if ! curl -fsS --max-time 120 "http://${HOST}/api/measurements/export" -o "$OUT"; then
    echo "!! could not export from ${HOST}." >&2
    echo "   Run this on the head (requests from the node itself need no sign-in)," >&2
    echo "   and check that AINode is running there: curl -s ${HOST}/api/health" >&2
    exit 1
fi

echo "==> wrote $OUT ($(du -h "$OUT" | cut -f1))"
if command -v python3 >/dev/null 2>&1; then
    python3 - "$OUT" <<'PY'
import json, sys

data = json.load(open(sys.argv[1]))
nodes = data.get("nodes", [])
print(f"    {len(nodes)} node(s), vLLM {data.get('engine_build', {}).get('ENGINE_VLLM_VERSION', '?')}")
print(f"    {'model':<52} {'node':<10} {'launches':>8} {'GB':>7} {'tok/s':>7}  launch")
for model in data.get("models", []):
    for row in model.get("measurements", []):
        launch = row.get("launch") or {}
        split = f"TP{launch.get('tensor_parallel_size', 1)}xPP{launch.get('pipeline_parallel_size', 1)}" \
            if launch else "-"
        window = launch.get("max_model_len") or row.get("max_model_len") or ""
        print(f"    {model['model'][:52]:<52} {str(row.get('node_id', ''))[:10]:<10} "
              f"{row.get('launches', 0):>8} {row.get('memory_gb') or 0:>7} "
              f"{row.get('tokens_per_second') or 0:>7}  {split} {window}")
PY
fi
