#!/usr/bin/env bash
# One command to bring the whole cluster to the current code.
#
#   scripts/update-cluster.sh --head 192.168.1.2 --nodes spark2,spark3
#
# Safe to run again, and again: every step checks what is already true before
# changing anything. Nothing here is specific to a version, so this is the
# command for every future update as well.
#
# What it does, in order:
#   1. checks it can reach every node before touching any of them
#   2. pulls the latest code
#   3. rebuilds the orchestrator image (the engine image only with --base)
#   4. optionally sets up the head's registry cache (--registry, idempotent)
#   5. distributes the new image — through the local registry when it exists,
#      otherwise by streaming it over SSH
#   6. restarts the service, members first, head last
#   7. verifies every node reports the new version
#
# Options:
#   --head <ip>        address of this node that the peers can reach (required
#                      for --registry; otherwise optional)
#   --nodes a,b,c      the other nodes, as SSH targets
#   --registry         set up / refresh the image cache on the head
#   --base             also rebuild the engine image (15-25 min)
#   --skip-pull        do not git pull (build what is checked out)
#   --skip-build       do not build (distribute and restart what exists)
#   --image NAME       orchestrator image tag (default: ainode:dev)
#   --check            say what would happen, change nothing
set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
cd "$REPO_ROOT"

HEAD_IP=""
NODES=""
IMAGE="${AINODE_IMAGE:-ainode:dev}"
DO_REGISTRY=0
DO_BASE=0
SKIP_PULL=0
SKIP_BUILD=0
CHECK=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --head) HEAD_IP="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --registry) DO_REGISTRY=1; shift ;;
        --base) DO_BASE=1; shift ;;
        --skip-pull) SKIP_PULL=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --check) CHECK=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
step() { printf '\n\033[1;36m── %s ─────────────────────────────\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }
run()  { if [[ $CHECK -eq 1 ]]; then printf '   would run: %s\n' "$*"; else "$@"; fi; }

# Root on another node. `ssh node "sudo ..."` fails with "sudo: a terminal is
# required" unless that node has passwordless sudo: ssh allocates no TTY for a
# command and sudo will not read a password without one. Try the
# non-interactive form, fall back to `ssh -t` so sudo can prompt.
remote_sudo() {
    local node="$1"; shift
    if [[ $CHECK -eq 1 ]]; then
        printf '   would run on %s: sudo %s\n' "$node" "$*"
        return 0
    fi
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "sudo -n true" 2>/dev/null; then
        ssh -o BatchMode=yes "$node" "sudo $*"
    else
        ssh -t "$node" "sudo $*"
    fi
}

NODE_LIST=()
if [[ -n "$NODES" ]]; then
    IFS=',' read -r -a NODE_LIST <<< "$NODES"
fi

# --- 1. preflight -----------------------------------------------------------
#
# Everything, before anything. A run that rebuilds the image and then finds it
# cannot reach the second node has left the cluster on two versions, which is
# the one state worse than the one it started in.

step "Checking the cluster"
command -v docker >/dev/null || die "docker is not installed here"
docker info >/dev/null 2>&1 || die "cannot talk to the Docker daemon (is your user in the docker group?)"
[[ -f pyproject.toml ]] || die "run this from the AINode repository"

VERSION_BEFORE="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
say "this node: AINode ${VERSION_BEFORE}, docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '?')"

for node in "${NODE_LIST[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" true 2>/dev/null \
        || die "cannot ssh to '${node}' without a password. Fix that first: ssh-copy-id ${node}"
    ssh -o BatchMode=yes "$node" "command -v docker >/dev/null" \
        || die "'${node}' has no docker"
    if ssh -o BatchMode=yes "$node" "sudo -n true" 2>/dev/null; then
        say "${node}: reachable, docker present, sudo without a password"
    else
        # Said now rather than discovered after a fifteen-minute build.
        warn "${node}: sudo will ask for a password — you will be prompted during the run"
        NEEDS_PASSWORD=1
        say "${node}: reachable, docker present"
    fi
done

if [[ $CHECK -eq 0 ]]; then
    # Ask for the local password once, at the start.
    sudo -n true 2>/dev/null || { warn "this node: sudo needs a password"; sudo -v; }
fi
if [[ ${NEEDS_PASSWORD:-0} -eq 1 ]]; then
    warn "Passwordless sudo on the peers makes this unattended:"
    warn "  echo \"\$USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl, /usr/bin/python3\" | sudo tee /etc/sudoers.d/ainode-update"
fi

# --- 2. code ----------------------------------------------------------------

step "Code"
if [[ $SKIP_PULL -eq 1 ]]; then
    say "skipping git pull (--skip-pull)"
else
    if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
        warn "the working tree has local changes; pulling anyway, git will stop if it conflicts"
    fi
    run git pull --ff-only
fi
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
say "building version ${VERSION}"

# --- 3. build ---------------------------------------------------------------

step "Build"
if [[ $SKIP_BUILD -eq 1 ]]; then
    say "skipping the build (--skip-build)"
else
    if [[ $DO_BASE -eq 1 ]]; then
        say "rebuilding the engine image — this takes 15-25 minutes"
        run scripts/build-base-image.sh
    else
        say "engine image left alone (pass --base to rebuild it)"
    fi
    run scripts/build-ainode-image.sh
fi
if [[ $CHECK -eq 0 ]]; then
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
        || die "${IMAGE} does not exist after the build"
fi

# --- 4. registry ------------------------------------------------------------

step "Image cache"
REGISTRY=""
if [[ $DO_REGISTRY -eq 1 ]]; then
    [[ -n "$HEAD_IP" ]] || die "--registry needs --head <ip> — the peers have to reach it somehow"
    args=(--head "$HEAD_IP")
    [[ -n "$NODES" ]] && args+=(--nodes "$NODES")
    [[ $CHECK -eq 1 ]] && args+=(--check)
    scripts/setup-registry-cache.sh "${args[@]}"
    REGISTRY="${HEAD_IP}:5001"
elif [[ -n "$HEAD_IP" ]] && curl -fsS --max-time 3 "http://${HEAD_IP}:5001/v2/" >/dev/null 2>&1; then
    # Already set up by an earlier run — use it without being asked twice.
    say "local registry at ${HEAD_IP}:5001 is up; using it"
    REGISTRY="${HEAD_IP}:5001"
else
    say "no local registry (pass --registry to set one up); images will stream over SSH"
fi

# --- 5. distribute ----------------------------------------------------------
#
# Through the registry when there is one: peers pull only the layers they are
# missing, so a version bump moves megabytes rather than a whole image. Over
# SSH otherwise, which always works and always moves everything.

step "Distributing ${IMAGE}"
if [[ ${#NODE_LIST[@]} -eq 0 ]]; then
    say "no peers given (--nodes); nothing to distribute"
elif [[ -n "$REGISTRY" ]]; then
    REMOTE_TAG="${REGISTRY}/${IMAGE}"
    say "pushing to ${REMOTE_TAG}"
    run docker tag "$IMAGE" "$REMOTE_TAG"
    run docker push "$REMOTE_TAG"
    for node in "${NODE_LIST[@]}"; do
        say "${node}: pulling"
        # Retagged locally, so every node ends up with the same plain name and
        # nothing else in the system has to know a registry exists.
        run ssh -o BatchMode=yes "$node" \
            "docker pull ${REMOTE_TAG} && docker tag ${REMOTE_TAG} ${IMAGE}"
    done
else
    for node in "${NODE_LIST[@]}"; do
        say "${node}: streaming the image over SSH"
        if [[ $CHECK -eq 1 ]]; then
            printf '   would run: docker save %s | ssh %s docker load\n' "$IMAGE" "$node"
        else
            docker save "$IMAGE" | ssh -o BatchMode=yes "$node" docker load
        fi
    done
fi

# --- 6. restart -------------------------------------------------------------
#
# Members first: the head forms the cluster from what it discovers, so it
# should come up last and see a complete one.

step "Restarting"
for node in "${NODE_LIST[@]}"; do
    say "${node}: restarting ainode"
    remote_sudo "$node" "systemctl restart ainode" \
        || warn "${node}: restart failed — check 'journalctl -u ainode -n 50' there"
done
say "this node: restarting ainode"
run sudo systemctl restart ainode

# --- 7. verify --------------------------------------------------------------

step "Verifying"
if [[ $CHECK -eq 1 ]]; then
    say "check mode — nothing was changed"
    exit 0
fi

check_node() {
    local label="$1" url="$2"
    for _ in $(seq 1 30); do
        local got
        got="$(curl -fsS --max-time 3 "$url/api/status" 2>/dev/null \
               | python3 -c 'import json,sys; print(json.load(sys.stdin).get("version",""))' 2>/dev/null || true)"
        if [[ -n "$got" ]]; then
            if [[ "$got" == "$VERSION" ]]; then
                say "${label}: ${got}"
            else
                warn "${label}: reports ${got}, expected ${VERSION}"
            fi
            return 0
        fi
        sleep 2
    done
    warn "${label}: did not answer within 60s — check 'journalctl -u ainode -n 50' there"
}

check_node "this node" "http://localhost:3000"
for node in "${NODE_LIST[@]}"; do
    host="$node"
    check_node "$node" "http://${host}:3000"
done

step "Done"
say "AINode ${VERSION} is deployed."
if [[ -n "$REGISTRY" ]]; then
    say "Next update: the same command. Only changed layers will move."
fi
