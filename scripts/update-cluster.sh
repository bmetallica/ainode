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
#   5b. distributes the ENGINE images too — after --base rebuilt them, or on
#      demand with --images: eugr's launcher aborts when the nodes disagree
#      about the one it uses, and an image-generation engine that exists only
#      on the head cannot serve on node 3
#   6. restarts the service, members first, head last
#   7. verifies every node reports the new version
#
# Options:
#   --head <ip>        address of this node that the peers can reach (required
#                      for --registry; otherwise optional)
#   --nodes a,b,c      the other nodes, as SSH targets
#   --registry         set up / refresh the image cache on the head
#   --base             also rebuild the engine image (15-25 min)
#   --images           distribute the engine images without rebuilding
#                      anything — for an image built here by hand, such as
#                      scripts/build-diffusers-image.sh
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
DO_IMAGES=0
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
        --images) DO_IMAGES=1; shift ;;
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

# Running inside the AINode container — which is where the UI's update button
# runs this. The docker socket and the SSH keys are mounted, so building,
# distributing and restarting the PEERS all work exactly as on the host. Two
# things do not: there is no sudo to prime, and no systemd to restart this
# node with. The head's own restart becomes a docker stop of this container,
# which the unit's Restart=always turns into a start on the new image.
IN_CONTAINER=0
if [[ "${AINODE_IN_CONTAINER:-}" == "1" || -f /.dockerenv ]]; then
    IN_CONTAINER=1
fi

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

if [[ $CHECK -eq 0 && $IN_CONTAINER -eq 0 ]]; then
    # Ask for the local password once, at the start.
    sudo -n true 2>/dev/null || { warn "this node: sudo needs a password"; sudo -v; }
fi
if [[ $IN_CONTAINER -eq 1 ]]; then
    say "this node: inside the AINode container — restarting through docker, not systemd"
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

# --- 5b. the engine image ---------------------------------------------------
#
# --base rebuilt it HERE. eugr's launcher compares image ids across the
# cluster before it starts anything and aborts when they differ:
#
#   Error: Cluster launch aborted because image 'vllm-node' is not in sync.
#
# which it is right to do — ranks on different builds fail later and far less
# clearly. But it left the cluster unable to launch anything until someone
# noticed, because a rebuild on one node is a divergence on the others.
#
# AINode's launch-time image distribution does not cover this one: it fires
# only for a model whose recipe PINS an engine image, on the reasoning that
# the launcher default is built locally on every node. That was true when
# every node ran this script. It stopped being true the moment the build
# moved to the head alone.

# The list, not one image. AINode grew a second engine — diffusers, for image
# generation — built locally by scripts/build-diffusers-image.sh and therefore
# in no registry to pull from. An engine image that exists only on the head
# cannot serve on node 3, which is where it is meant to run.
ENGINE_IMAGES=("${ENGINE_IMAGE:-vllm-node:latest}")
for extra in "${DIFFUSERS_IMAGE:-ainode-diffusers:latest}"; do
    # Only what is actually here: a cluster that never built the image
    # generation engine should not be told it is missing one.
    if docker image inspect "$extra" >/dev/null 2>&1; then
        ENGINE_IMAGES+=("$extra")
    fi
done

if [[ ( $DO_BASE -eq 1 || $DO_IMAGES -eq 1 ) && ${#NODE_LIST[@]} -gt 0 ]]; then
  for ENGINE_IMAGE in "${ENGINE_IMAGES[@]}"; do
    step "Distributing ${ENGINE_IMAGE}"
    if ! docker image inspect "$ENGINE_IMAGE" >/dev/null 2>&1; then
        warn "${ENGINE_IMAGE} is not here; skipping (did the build fail?)"
    elif [[ -n "$REGISTRY" ]]; then
        ENGINE_REMOTE="${REGISTRY}/${ENGINE_IMAGE}"
        say "pushing ${ENGINE_IMAGE} to ${ENGINE_REMOTE} (~20 GB on the first run)"
        run docker tag "$ENGINE_IMAGE" "$ENGINE_REMOTE"
        run docker push "$ENGINE_REMOTE"
        for node in "${NODE_LIST[@]}"; do
            say "${node}: pulling the engine image"
            run ssh -o BatchMode=yes "$node" \
                "docker pull ${ENGINE_REMOTE} && docker tag ${ENGINE_REMOTE} ${ENGINE_IMAGE}"
        done
    else
        for node in "${NODE_LIST[@]}"; do
            say "${node}: streaming ${ENGINE_IMAGE} over SSH (slow)"
            if [[ $CHECK -eq 1 ]]; then
                printf '   would run: docker save %s | ssh %s docker load\n' \
                    "$ENGINE_IMAGE" "$node"
            else
                docker save "$ENGINE_IMAGE" | ssh -o BatchMode=yes "$node" docker load
            fi
        done
    fi

    # The launcher compares ids, not tags. Say so here rather than letting the
    # next launch be the thing that reports it.
    if [[ $CHECK -eq 0 ]]; then
        head_id="$(docker image inspect --format '{{.Id}}' "$ENGINE_IMAGE" 2>/dev/null || true)"
        for node in "${NODE_LIST[@]}"; do
            peer_id="$(ssh -o BatchMode=yes "$node" \
                "docker image inspect --format '{{.Id}}' ${ENGINE_IMAGE}" 2>/dev/null || true)"
            if [[ -n "$head_id" && "$peer_id" == "$head_id" ]]; then
                say "${node}: ${ENGINE_IMAGE} in sync"
            else
                warn "${node}: ${ENGINE_IMAGE} still differs — a launch there will abort"
            fi
        done
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
if [[ $IN_CONTAINER -eq 0 ]]; then
    say "this node: restarting ainode"
    run sudo systemctl restart ainode
else
    # Deferred to the very end: stopping this container ends this script, so
    # nothing after it would run — including the verification below.
    say "this node: restart deferred to the end (it stops the container running this)"
fi

# --- 7. verify --------------------------------------------------------------

step "Verifying"
if [[ $CHECK -eq 1 ]]; then
    say "check mode — nothing was changed"
    exit 0
fi

# Asked over SSH rather than over HTTP by name. "Spark2" is an SSH alias — it
# is in ~/.ssh/config, not necessarily in DNS or /etc/hosts, so
# http://Spark2:3000 resolves to nothing and every member "did not answer",
# on a cluster where every member was in fact answering. Asking the node to
# curl its own localhost sidesteps naming and any firewall between us.
check_node() {
    local label="$1" node="$2"
    local probe='curl -fsS --max-time 3 http://localhost:3000/api/status'
    for _ in $(seq 1 30); do
        local got payload
        if [[ -z "$node" ]]; then
            payload="$(eval "$probe" 2>/dev/null || true)"
        else
            payload="$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "$probe" 2>/dev/null || true)"
        fi
        got="$(printf '%s' "$payload" \
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

if [[ $IN_CONTAINER -eq 0 ]]; then
    check_node "this node" ""
fi
for node in "${NODE_LIST[@]}"; do
    check_node "$node" "$node"
done

step "Done"
say "AINode ${VERSION} is deployed."
if [[ -n "$REGISTRY" ]]; then
    say "Next update: the same command. Only changed layers will move."
fi

# --- 8. this node, last -----------------------------------------------------
#
# Only in the container, and only now: everything above still needed this
# process alive.
if [[ $IN_CONTAINER -eq 1 && $CHECK -eq 0 ]]; then
    step "Restarting this node"
    if [[ "${AINODE_UNIT_SWAPPABLE:-}" != "1" ]]; then
        # The old unit either pins its image or does not restart on a clean
        # exit, so stopping ourselves would drop the head rather than swap it.
        warn "this container was not started by a swappable unit — restart the head yourself:"
        warn "    sudo systemctl restart ainode"
        exit 0
    fi
    # The unit launches whatever image.env names, and reads it at every start.
    # Without this the build above is discarded: systemd brings back the image
    # the node was already running, and an update that changed nothing looks
    # exactly like one that worked.
    HOME_DIR="${AINODE_HOME:-/root/.ainode}"
    if [[ -d "$HOME_DIR" ]] && ! grep -qx "AINODE_IMAGE=${IMAGE}" "${HOME_DIR}/image.env" 2>/dev/null; then
        say "pinning ${IMAGE} for the service to start"
        printf 'AINODE_IMAGE=%s\n' "$IMAGE" > "${HOME_DIR}/image.env.tmp"
        mv -f "${HOME_DIR}/image.env.tmp" "${HOME_DIR}/image.env"
    fi
    say "stopping this container; systemd starts it again on ${IMAGE}"
    exec docker stop -t 30 ainode
fi
