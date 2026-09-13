#!/usr/bin/env bash
# Collect everything needed to diagnose a launch, in one paste-able report.
#
#   scripts/diagnose.sh > report.txt
#   scripts/diagnose.sh --nodes Spark2,Spark3 > report.txt
#
# Read-only: it inspects, it never stops, removes or restarts anything.
#
# Secrets are redacted — Hugging Face tokens, API keys, passwords, the cluster
# secret. Check the output before sending it anyway; a report that has to be
# trusted blindly is a report nobody should paste.
set -uo pipefail          # deliberately NOT -e: a failing probe must not end
                          # the report, it is itself a finding

NODES=""
LOG_LINES="${AINODE_DIAG_LOG_LINES:-120}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes) NODES="${2:-}"; shift 2 ;;
        --log-lines) LOG_LINES="${2:-120}"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

CONTAINER="${AINODE_CONTAINER:-ainode}"
WEB_PORT="${AINODE_WEB_PORT:-3000}"

# --- redaction --------------------------------------------------------------
#
# Everything printed goes through this. An operator pasting a report should not
# have to think about what is in it — a token has already been pasted into a
# chat once in this project's life, and that is once more than necessary.
redact() {
    sed -E \
        -e 's/hf_[A-Za-z0-9]{16,}/hf_***REDACTED***/g' \
        -e 's/(sk-|ghp_|gho_|github_pat_|xoxb-|nvapi-)[A-Za-z0-9_-]{16,}/\1***REDACTED***/g' \
        -e 's/("?(hf_token|huggingface_token|token|api_key|apikey|password|secret|cluster_secret|mqtt_password)"?[[:space:]]*[:=][[:space:]]*"?)[^",}[:space:]]+/\1***REDACTED***/gI' \
        -e 's/(Authorization:[[:space:]]*(Bearer|Basic)[[:space:]]+)[^[:space:]]+/\1***REDACTED***/gI' \
        -e 's/(HUGGING_FACE_HUB_TOKEN|HF_TOKEN|AINODE_API_KEY)=[^[:space:]]+/\1=***REDACTED***/g'
}

# Probes fail — that is data, not a crash. These read stdin and say so
# plainly instead of printing a Python traceback into the report.
read -r -d '' PY_STATUS <<'PYEOF' || true
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("(no answer from the API)"); raise SystemExit
try:
    d = json.loads(raw)
except Exception:
    print("(not JSON):", raw[:200]); raise SystemExit
keys = ("version", "node_id", "node_name", "model", "engine_ready",
        "load_phase", "load_detail", "load_error", "cluster_role",
        "distributed_mode")
print(json.dumps({k: d.get(k) for k in keys}, indent=1))
PYEOF

read -r -d '' PY_NODES <<'PYEOF' || true
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("(no answer from the API)"); raise SystemExit
try:
    nodes = json.loads(raw).get("nodes", [])
except Exception:
    print("(not JSON):", raw[:200]); raise SystemExit
for n in nodes:
    print(n.get("node_name") or n.get("node_id"), "|", n.get("status"),
          "| model:", n.get("model") or "-",
          "| ready:", n.get("engine_ready"))
    for i in n.get("instances") or []:
        print("    ", i.get("model"), "port", i.get("api_port"),
              "status", i.get("status"), "phase", i.get("load_phase") or "-",
              "detail", i.get("load_detail") or "-")
        if i.get("load_error"):
            print("      error:", str(i["load_error"])[:400])
PYEOF

read -r -d '' PY_PROFILES <<'PYEOF' || true
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("(no profiles.json)"); raise SystemExit
try:
    d = json.loads(raw)
except Exception:
    print("(not JSON)"); raise SystemExit
print("default:", d.get("default") or "-")
for p in d.get("profiles") or []:
    print(" ", p.get("name"), len(p.get("entries") or []), "entries")
PYEOF

read -r -d '' PY_PRETTY <<'PYEOF' || true
import json, sys
raw = sys.stdin.read().strip()
if not raw:
    print("(empty)"); raise SystemExit
try:
    print(json.dumps(json.loads(raw), indent=1)[:6000])
except Exception:
    print(raw[:2000])
PYEOF

section() { printf '\n===== %s =====\n' "$*"; }
run()     { printf '$ %s\n' "$*"; eval "$@" 2>&1 | redact; }

printf '# AINode diagnostic report\n'
printf '# generated %s on %s\n' "$(date -Is)" "$(hostname)"
printf '# secrets redacted; read before sending\n'

# --- 1. what is running -----------------------------------------------------

section "Versions"
run "docker version --format '{{.Server.Version}}'"
run "uname -srm"
run "curl -fsS --max-time 5 http://localhost:${WEB_PORT}/api/status | python3 -c \"\$PY_STATUS\""

section "Hardware"
run "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv"
run "free -g"
run "df -h / \$HOME"

section "Service"
run "systemctl is-active ainode"
run "systemctl show ainode -p ExecStart --value | head -c 1200"
run "docker ps -a --filter name=ainode --filter name=vllm_node --filter name=registry --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'"

# --- 2. the cluster's own view ----------------------------------------------

section "Cluster"
run "curl -fsS --max-time 5 http://localhost:${WEB_PORT}/api/cluster/resources | python3 -c \"\$PY_PRETTY\""

section "Instances (this node)"
run "curl -fsS --max-time 5 http://localhost:${WEB_PORT}/api/nodes | python3 -c \"\$PY_NODES\""

# --- 3. what the engine was actually told -----------------------------------

section "Generated launch scripts"
run "docker exec ${CONTAINER} sh -c 'ls -la /opt/spark-vllm-docker/examples/ainode-*.sh 2>/dev/null; for f in /opt/spark-vllm-docker/examples/ainode-*.sh; do [ -f \"\$f\" ] && { echo; echo \"--- \$f\"; cat \"\$f\"; }; done'"

section "Engine container"
run "docker inspect vllm_node --format '{{.Config.Image}} | started {{.State.StartedAt}} | running={{.State.Running}}' "
run "docker inspect vllm_node --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{\"\\n\"}}{{end}}'"
run "docker inspect vllm_node --format '{{range .Config.Env}}{{.}}{{\"\\n\"}}{{end}}' | grep -E 'NCCL|VLLM|HF_|CUDA|B12X|INSTANTTENSOR' | head -40"

section "Images"
run "docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}' | grep -Ei 'vllm|ainode|registry' | head -20"

# --- 4. models on disk ------------------------------------------------------

section "Models on disk"
run "docker exec ${CONTAINER} sh -c 'ls -la /root/.ainode/models | head -30'"
run "docker exec ${CONTAINER} sh -c 'du -sh /root/.ainode/models/* 2>/dev/null | sort -h | tail -15'"

# --- 5. configuration -------------------------------------------------------

section "Config (redacted)"
run "docker exec ${CONTAINER} sh -c 'cat /root/.ainode/config.json 2>/dev/null' | python3 -c \"\$PY_PRETTY\""
run "docker exec ${CONTAINER} sh -c 'cat /root/.ainode/instances.json 2>/dev/null' | head -c 2000"
run "docker exec ${CONTAINER} sh -c 'cat /root/.ainode/profiles.json 2>/dev/null' | python3 -c \"\$PY_PROFILES\""

# --- 6. logs ----------------------------------------------------------------
#
# Two passes per log: the lines that carry meaning (timings, phases, errors)
# from the whole file, then the raw tail. The interesting lines are usually
# thousands of lines before the end, behind a wall of throughput counters.

log_report() {
    local path="$1" label="$2"
    section "Log: ${label} — signal lines"
    run "docker exec ${CONTAINER} sh -c 'grep -nE \"Starting to load model|Loading model from scratch|Loading weights|Model loading took|loading took|torch.compile|Capturing CUDA graph|graph capturing finished|init engine|Available KV cache|Application startup complete|Uvicorn running|error|Error|ERROR|Traceback|RuntimeError|unrecognized|Killed|OutOfMemory|CUDA\" ${path} 2>/dev/null | grep -vE \"Avg prompt throughput|loggers.py\" | tail -${LOG_LINES}' | cut -c1-400"
    section "Log: ${label} — tail"
    run "docker exec ${CONTAINER} sh -c 'tail -40 ${path} 2>/dev/null' | cut -c1-400"
}

log_report /root/.ainode/logs/vllm.log "vllm.log (solo)"
log_report /root/.ainode/logs/distributed.log "distributed.log"

section "Orchestrator log"
run "docker logs --tail 80 ${CONTAINER} 2>&1 | cut -c1-400"

section "Journal"
run "journalctl -u ainode -n 40 --no-pager 2>/dev/null | cut -c1-400"

# --- 7. peers ---------------------------------------------------------------
#
# A short version from each peer: enough to see whether it agrees about
# versions, images and what it is running.

if [[ -n "$NODES" ]]; then
    IFS=',' read -r -a NODE_LIST <<< "$NODES"
    for node in "${NODE_LIST[@]}"; do
        [[ -n "$node" ]] || continue
        section "Peer: ${node}"
        run "ssh -o BatchMode=yes -o ConnectTimeout=8 ${node} '
            echo \"--- status\"
            curl -fsS --max-time 5 http://localhost:${WEB_PORT}/api/status | head -c 900
            echo
            echo \"--- containers\"
            docker ps -a --filter name=ainode --filter name=vllm_node --format \"{{.Names}} {{.Image}} {{.Status}}\"
            echo \"--- images\"
            docker images --format \"{{.Repository}}:{{.Tag}} {{.ID}}\" | grep -Ei \"vllm|ainode\" | head -8
            echo \"--- engine log tail\"
            docker exec ainode sh -c \"tail -25 /root/.ainode/logs/vllm.log 2>/dev/null\" | cut -c1-300
        '"
    done
fi

printf '\n===== end of report =====\n'
