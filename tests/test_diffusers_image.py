"""The image-generation engine image, and how it reaches the other nodes.

An engine image that exists only on the head is worth nothing on node 3,
which is where it is meant to run. That was the gap in the first draft of
images.md: the plan said what to build and not how it travels.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = (ROOT / "scripts" / "Dockerfile.diffusers").read_text()
BUILD = (ROOT / "scripts" / "build-diffusers-image.sh").read_text()
UPDATE = (ROOT / "scripts" / "update-cluster.sh").read_text()


class TestItBuildsOnTheProvenStack:
    def test_it_starts_from_the_engine_image(self):
        # The hard part on this hardware is not diffusers, it is torch with
        # CUDA for aarch64 and Blackwell — and that is already built, already
        # proven, and already on every node.
        assert "ARG ENGINE_BASE=vllm-node:latest" in DOCKERFILE
        assert "FROM ${ENGINE_BASE}" in DOCKERFILE

    def test_torch_is_not_replaced_underneath(self):
        # diffusers lists torch as a dependency, and a plain install happily
        # swaps the aarch64/Blackwell build for a generic wheel — which is
        # the one failure this image exists to avoid.
        assert "--no-cache-dir" in DOCKERFILE
        install = DOCKERFILE.split("pip install")[1].split("&&")[0]
        assert "torch" not in install

    def test_the_build_is_verified_in_the_image(self):
        # A build that produces an image where `import diffusers` fails is a
        # build that should not have succeeded.
        assert "import torch, diffusers" in DOCKERFILE

    def test_the_build_script_refuses_without_the_base(self):
        assert "docker image inspect" in BUILD
        assert "scripts/build-base-image.sh" in BUILD

    def test_the_build_script_says_what_comes_next(self):
        # Building it and stopping there is the mistake the plan called out.
        assert "update-cluster.sh --images" in BUILD


class TestItTravels:
    def test_the_distribution_takes_a_list_not_one_image(self):
        assert "ENGINE_IMAGES=(" in UPDATE
        assert 'for ENGINE_IMAGE in "${ENGINE_IMAGES[@]}"' in UPDATE

    def test_the_diffusers_image_is_in_it(self):
        assert "ainode-diffusers:latest" in UPDATE

    def test_an_image_that_was_never_built_is_not_missed(self):
        # A cluster that does not do image generation should not be told it
        # is missing an engine.
        block = UPDATE.split("ENGINE_IMAGES=(")[1][:600]
        assert "docker image inspect" in block

    def test_it_can_be_distributed_without_rebuilding_anything(self):
        # The image is built by hand on the head; forcing --base (15-25
        # minutes of vLLM) to move it would be absurd.
        assert "--images) DO_IMAGES=1" in UPDATE
        assert "DO_BASE -eq 1 || $DO_IMAGES -eq 1" in UPDATE

    def test_both_routes_survive(self):
        # Registry where there is one, docker save over SSH where there is
        # not. Both were already there; both must still take the image name
        # from the loop rather than a fixed one.
        block = UPDATE.split('for ENGINE_IMAGE in "${ENGINE_IMAGES[@]}"')[1]
        assert "docker push" in block
        assert "docker save" in block

    def test_the_id_comparison_names_the_image(self):
        # Tags lie and ids do not — and with two images in play, "engine
        # image still differs" would no longer say which.
        assert '${ENGINE_IMAGE} still differs' in UPDATE

    def test_the_flag_is_documented_in_the_header(self):
        header = UPDATE.split("set -euo pipefail")[0]
        assert "--images" in header
