#!/usr/bin/env bash
# Build the image-generation engine image.
#
#   scripts/build-diffusers-image.sh
#
# FROM the vLLM engine image (vllm-node:latest by default), so the torch and
# CUDA build proven on this hardware is the one diffusers runs on. Build the
# base first if it is not here: scripts/build-base-image.sh
#
# Then distribute it — an image on the head alone is worth nothing on node 3:
#   scripts/update-cluster.sh --images --nodes Spark2,Spark3
set -euo pipefail

REPO_ROOT="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"
cd "$REPO_ROOT"

ENGINE_BASE="${ENGINE_BASE:-vllm-node:latest}"
IMAGE="${DIFFUSERS_IMAGE:-ainode-diffusers:latest}"
# What to install for diffusers. A released version is right for most
# checkpoints; one written against an unreleased diffusers needs a git ref,
# which is not unusual for a model published the week its architecture lands:
#
#   DIFFUSERS_REF="git+https://github.com/huggingface/diffusers@main" \
#     scripts/build-diffusers-image.sh
#
# The symptom without it is "module diffusers has no attribute
# QwenImage21Pipeline" — the class named in the checkpoint's model_index.json
# does not exist in the library that is installed.
DIFFUSERS_REF="${DIFFUSERS_REF:-diffusers>=0.36}"

if ! docker image inspect "$ENGINE_BASE" >/dev/null 2>&1; then
    echo "The engine base image ${ENGINE_BASE} is not on this node." >&2
    echo "Build it first: scripts/build-base-image.sh" >&2
    exit 1
fi

echo "==> Building ${IMAGE} FROM ${ENGINE_BASE}"
echo "    diffusers: ${DIFFUSERS_REF}"
docker build \
    --build-arg "ENGINE_BASE=${ENGINE_BASE}" \
    --build-arg "DIFFUSERS_REF=${DIFFUSERS_REF}" \
    -f scripts/Dockerfile.diffusers \
    -t "$IMAGE" \
    scripts/

echo "==> Built ${IMAGE}"
docker image inspect --format '    id:   {{.Id}}' "$IMAGE"
docker image inspect --format '    size: {{.Size}} bytes' "$IMAGE"
echo
echo "==> Next: put it on the other nodes, or nothing can launch there."
echo "    scripts/update-cluster.sh --images --nodes Spark2,Spark3"
