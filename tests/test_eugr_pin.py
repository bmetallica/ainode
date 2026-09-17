"""The launcher in our image and the engine it drives come from one commit.

Both pins said c026c92 (2026-04-13) while the engine images actually in
service were built from a September checkout — 207 upstream commits and 644
lines of launch-cluster.sh later. The pins agreed with each other and with
nothing that was running.

The gap was not theoretical. The April launcher copies its start script to
/workspace WITHOUT creating the directory first — `ensure_container_workspace`
does not exist in it — so a distributed launch against any image lacking
/workspace died with

    Error response from daemon: Could not find the file /workspace in
    container vllm_node

which reads as a fault in the engine image, and was diagnosed as one.

A stale pin is worse than no pin: it looks deliberate.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUILD_SH = (REPO / "scripts" / "build-base-image.sh").read_text()
DOCKERFILE = (REPO / "scripts" / "Dockerfile.ainode").read_text()

_SHA = re.compile(r"\b[0-9a-f]{40}\b")


def _pin(text: str, marker: str) -> str:
    line = next(ln for ln in text.splitlines()
                if ln.startswith(marker) and _SHA.search(ln))
    return _SHA.search(line).group(0)


class TestTheTwoPinsAgree:
    def test_the_builder_and_the_image_name_the_same_commit(self):
        """The launcher baked into our image and the engine it drives have to
        come from one upstream tree. A launcher that predates the image's
        conventions fails in ways that look like the image's fault."""
        assert _pin(BUILD_SH, "EUGR_COMMIT=") == _pin(DOCKERFILE, "ARG EUGR_COMMIT=")

    def test_the_pin_is_not_the_stale_april_one(self):
        """Kept as a named regression: five months of upstream work, including
        the /workspace fix, sat behind this value."""
        stale = "c026c92bd0c1236f947ac212565b15a33ba1b4e7"
        assert _pin(BUILD_SH, "EUGR_COMMIT=") != stale
        assert _pin(DOCKERFILE, "ARG EUGR_COMMIT=") != stale

    def test_bumping_it_is_documented_as_deliberate(self):
        """Upstream's Dockerfile carries hardware-specific workarounds — the
        DeepGEMM ref is pinned BACK over an SM121 DeepSeek-V4 regression — so
        moving forward is not automatically moving up."""
        assert "read the commits first" in BUILD_SH
