#!/usr/bin/env bash
# Set up the head's image cache, and point every node at it.
#
# Two registries, because one container cannot be both:
#
#   :5000  pull-through cache for Docker Hub. Fetches a layer from the
#          internet once and serves it to the other nodes over the local
#          network. This is where the ~20 GB engine images stop being
#          downloaded once per node.
#   :5001  a plain registry for images built here (ainode, vllm-node), which
#          exist in no public registry. Peers pull only the layers they are
#          missing, so an update moves megabytes instead of a whole image.
#
# Idempotent: run it as often as you like. It creates what is missing, leaves
# what is correct alone, and restarts the Docker daemon ONLY on nodes whose
# configuration actually changed.
#
# Usage:
#   scripts/setup-registry-cache.sh --head 192.168.1.2 --nodes spark2,spark3
#   scripts/setup-registry-cache.sh --head 192.168.1.2 --nodes spark2 --check
#
# --check reports what it would change and touches nothing.
set -euo pipefail

HEAD_IP=""
NODES=""
CHECK=0
CACHE_DIR="${AINODE_REGISTRY_DIR:-/var/lib/ainode-registry}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --head) HEAD_IP="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --check) CHECK=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[[ -n "$HEAD_IP" ]] || die "--head <ip> is required: the peers need an address that reaches this node."

# The daemon.json edit, as a program rather than a sed line: the file may
# already carry settings this has no business touching, so it is parsed,
# merged and written back only when something actually changed. Exit code 10
# means "changed", 0 means "already correct" — the caller uses that to decide
# whether to restart Docker.
read -r -d '' MERGE_PY <<'PY' || true
import json, os, sys

# Overridable so the merge can be exercised against a scratch file — the
# logic that rewrites a host's Docker configuration is worth a test.
path = os.environ.get("DAEMON_JSON", "/etc/docker/daemon.json")
mirror = sys.argv[1]
insecure = sys.argv[2]
local = sys.argv[3]

try:
    with open(path) as fh:
        text = fh.read().strip()
    config = json.loads(text) if text else {}
except FileNotFoundError:
    config = {}
except json.JSONDecodeError:
    print(f"{path} is not valid JSON; refusing to touch it", file=sys.stderr)
    raise SystemExit(2)

before = json.dumps(config, sort_keys=True)

mirrors = list(config.get("registry-mirrors") or [])
if mirror not in mirrors:
    mirrors.insert(0, mirror)
config["registry-mirrors"] = mirrors

insecures = list(config.get("insecure-registries") or [])
for entry in (insecure, local):
    if entry not in insecures:
        insecures.append(entry)
config["insecure-registries"] = insecures

if json.dumps(config, sort_keys=True) == before:
    raise SystemExit(0)

if os.environ.get("CHECK_ONLY") == "1":
    print("would add: " + mirror + " and " + local)
    raise SystemExit(10)

os.makedirs("/etc/docker", exist_ok=True)
if os.path.exists(path):
    # One backup per change, timestamped: a daemon.json is a thing people
    # have edited by hand, and the copy is worth more than the tidiness.
    import shutil, time
    shutil.copy2(path, path + ".ainode-" + time.strftime("%Y%m%d-%H%M%S"))
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(config, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
print("updated " + path)
raise SystemExit(10)
PY

MIRROR="http://${HEAD_IP}:5000"
INSECURE="${HEAD_IP}:5000"
LOCAL="${HEAD_IP}:5001"

# --- 1. the two registries, on the head ------------------------------------

start_registry() {
    local name="$1" port="$2" proxy="$3"
    if docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q .; then
        say "${name} already running on :${port}"
        return 0
    fi
    if docker ps -a --filter "name=^${name}$" --format '{{.Names}}' | grep -q .; then
        say "starting existing ${name}"
        docker start "${name}" >/dev/null
        return 0
    fi
    if [[ $CHECK -eq 1 ]]; then
        warn "would create ${name} on :${port}"
        return 0
    fi
    say "creating ${name} on :${port}"
    sudo mkdir -p "${CACHE_DIR}/${name}"
    local args=(-d --restart=always --name "${name}" -p "${port}:5000"
                -v "${CACHE_DIR}/${name}:/var/lib/registry")
    if [[ -n "$proxy" ]]; then
        args+=(-e "REGISTRY_PROXY_REMOTEURL=${proxy}")
    else
        # Deleting a tag is how an update reclaims space from the version it
        # replaced; without this the registry refuses and only grows.
        args+=(-e "REGISTRY_STORAGE_DELETE_ENABLED=true")
    fi
    docker run "${args[@]}" registry:2 >/dev/null
}

start_registry ainode-registry-cache 5000 "https://registry-1.docker.io"
start_registry ainode-registry-local 5001 ""

# --- 2. point every node at them -------------------------------------------

configure_node() {
    local target="$1"          # "" = this node
    local label="${target:-this node}"
    local changed=0 rc=0

    if [[ -z "$target" ]]; then
        CHECK_ONLY="$CHECK" sudo -E python3 -c "$MERGE_PY" "$MIRROR" "$INSECURE" "$LOCAL" || rc=$?
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" \
            "CHECK_ONLY=$CHECK sudo -E python3 -c $(printf '%q' "$MERGE_PY") \
             $(printf '%q' "$MIRROR") $(printf '%q' "$INSECURE") $(printf '%q' "$LOCAL")" || rc=$?
    fi

    case $rc in
        0)  say "${label}: docker already points at the cache" ;;
        10) changed=1 ;;
        *)  die "${label}: could not update /etc/docker/daemon.json (rc=$rc)" ;;
    esac

    [[ $changed -eq 1 ]] || return 0
    if [[ $CHECK -eq 1 ]]; then
        warn "${label}: daemon.json WOULD change (Docker restart needed)"
        return 0
    fi

    # Restarting the daemon stops containers without a restart policy — the
    # AINode service among them — so the service is brought back deliberately
    # rather than left to chance.
    say "${label}: restarting Docker (containers will bounce)"
    if [[ -z "$target" ]]; then
        sudo systemctl restart docker
        sleep 3
        sudo systemctl restart ainode 2>/dev/null || true
    else
        ssh -o BatchMode=yes "$target" \
            "sudo systemctl restart docker && sleep 3 && (sudo systemctl restart ainode || true)"
    fi
}

configure_node ""
if [[ -n "$NODES" ]]; then
    IFS=',' read -r -a NODE_LIST <<< "$NODES"
    for node in "${NODE_LIST[@]}"; do
        [[ -n "$node" ]] && configure_node "$node"
    done
fi

# --- 3. show that it works --------------------------------------------------

if [[ $CHECK -eq 0 ]]; then
    say "cache contents:"
    curl -fsS "http://${HEAD_IP}:5000/v2/_catalog" 2>/dev/null || warn "cache not answering yet"
    echo
    curl -fsS "http://${HEAD_IP}:5001/v2/_catalog" 2>/dev/null || warn "local registry not answering yet"
    echo
fi

say "done"
