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

echo "==> Building orchestrator image ainode:${TAG} (context: $REPO_ROOT)"
docker build -f scripts/Dockerfile.ainode -t "ainode:${TAG}" .

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
