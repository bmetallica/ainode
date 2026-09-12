#!/usr/bin/env bash
# Fetch the experimental B12X engine image and distribute it.
#
# Some models run only on this stack — GLM 5.3 Flash is the one in our catalog
# — and its flags (--attention-backend B12X, --moe-backend b12x, --load-format
# b12x) exist in no other image. Running them against the default image fails
# in argparse with an exit code 2 that says nothing about the image.
#
# This is a SEPARATE script rather than a flag on build-base-image.sh because
# upstream's --exp-b12x is incompatible with the prebuilt-wheel path that one
# is built around (its own words: "B12X vLLM wheels are not published"). It
# pulls a prebuilt image rather than compiling, so it is minutes, not half an
# hour.
#
# Usage:
#   scripts/build-b12x-image.sh                      # fetch and tag locally
#   scripts/build-b12x-image.sh --nodes Spark2,Spark3   # and place it on peers
#
# The image is EXPERIMENTAL upstream. We have not served a model on it.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORKTREE="$SCRIPT_DIR/_eugr"
EUGR_REPO="${EUGR_REPO:-https://github.com/eugr/spark-vllm-docker.git}"
# The same commit the base image is built from: two engine images from
# different launcher generations would disagree about the .env contract.
EUGR_COMMIT="${EUGR_COMMIT:-$(sed -n 's/^EUGR_COMMIT="${EUGR_COMMIT:-\(.*\)}"/\1/p' "$SCRIPT_DIR/build-base-image.sh" | head -1)}"
IMAGE="${AINODE_B12X_IMAGE:-vllm-node-b12x}"
NODES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes) NODES="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker is not installed here"
[[ -n "$EUGR_COMMIT" ]] || die "could not read EUGR_COMMIT from build-base-image.sh"

say "eugr @ ${EUGR_COMMIT:0:8}"
if [[ -d "$WORKTREE/.git" ]]; then
    git -C "$WORKTREE" fetch --depth=1 origin "$EUGR_COMMIT" 2>/dev/null || true
    git -C "$WORKTREE" checkout -q "$EUGR_COMMIT"
else
    git clone --depth=1 "$EUGR_REPO" "$WORKTREE"
    git -C "$WORKTREE" fetch --depth=50 origin "$EUGR_COMMIT" || \
        git -C "$WORKTREE" fetch --unshallow
    git -C "$WORKTREE" checkout -q "$EUGR_COMMIT"
fi

# No Dockerfile patching here. The NCCL re-pin and the wheel override in
# build-base-image.sh both target the source build; --exp-b12x takes a
# published image and neither applies.
say "fetching the B12X image (prebuilt upstream; minutes, not a compile)"
pushd "$WORKTREE" >/dev/null
./build-and-copy.sh --exp-b12x --tag "$IMAGE"
popd >/dev/null

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "$IMAGE is not here after the fetch — read the output above"
docker tag "$IMAGE" "${IMAGE}:latest" 2>/dev/null || true
say "tagged ${IMAGE}:latest"

if [[ -n "$NODES" ]]; then
    IFS=',' read -r -a NODE_LIST <<< "$NODES"
    for node in "${NODE_LIST[@]}"; do
        [[ -n "$node" ]] || continue
        # The launcher compares image IDs across nodes, so every node has to
        # end up with this exact build, not merely something of the same name.
        say "${node}: placing the image"
        docker save "${IMAGE}:latest" | ssh -o BatchMode=yes "$node" docker load
    done
fi

say "done. A model needing it names it in the catalog (engine_image_eugr),"
say "or set Engine image to '${IMAGE}' in the launch panel."
