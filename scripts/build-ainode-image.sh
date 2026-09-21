#!/usr/bin/env bash
# Build the AINode orchestrator image.
#
# A one-line wrapper around:
#
#     docker build -f scripts/Dockerfile.ainode -t ainode:<tag> .
#
# It exists because that trailing "." is the build context, it is the easiest
# character in the command to lose when copying, and dropping it fails with
# "docker buildx build requires 1 argument" — a message that says nothing
# about a missing dot.
#
# This builds the ORCHESTRATOR (slim Python: web UI, API, cluster logic). The
# ENGINE image (CUDA, vLLM, NCCL) is a separate build — scripts/build-base-image.sh.
# The orchestrator does not build on the engine; it launches it at runtime.
#
# Usage:
#   scripts/build-ainode-image.sh             # tags ainode:<version> and ainode:dev
#   scripts/build-ainode-image.sh mytag       # tags ainode:mytag
set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
cd "$REPO_ROOT"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
TAG="${1:-$VERSION}"

# The image bakes in eugr's launcher, which must match the commit the ENGINE
# image was built from — they hand each other a .env and a launch script.
# Read it from the one place it is defined rather than repeating the SHA.
EUGR_COMMIT="$(sed -n 's/^EUGR_COMMIT="\${EUGR_COMMIT:-\(.*\)}"/\1/p' \
    scripts/build-base-image.sh | head -1)"
if [ -z "$EUGR_COMMIT" ]; then
    echo "!! could not read EUGR_COMMIT from scripts/build-base-image.sh" >&2
    echo "   The launcher pin would silently fall back to the Dockerfile default." >&2
    exit 1
fi

echo "==> Building orchestrator image ainode:${TAG} (context: $REPO_ROOT)"
echo "    eugr launcher pinned to ${EUGR_COMMIT:0:7}"
docker build -f scripts/Dockerfile.ainode \
    --build-arg "EUGR_COMMIT=${EUGR_COMMIT}" \
    --build-arg "AINODE_GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)" \
    -t "ainode:${TAG}" .

# Also tag :dev when building the release version — the installer examples and
# the mesh guide both use ainode:dev, and having it point at the latest local
# build is what an operator means by it.
if [ "$TAG" = "$VERSION" ]; then
    docker tag "ainode:${TAG}" "ainode:dev"
    echo "==> Tagged: ainode:${TAG} and ainode:dev"
else
    echo "==> Tagged: ainode:${TAG}"
fi

cat <<DONE

Install against it on this node:
    AINODE_IMAGE=ainode:${TAG} bash scripts/install.sh

Copy both images to the other nodes (engine first — it is the large one):
    docker save vllm-node:latest | ssh <peer> docker load
    docker save ainode:${TAG}    | ssh <peer> docker load
DONE
