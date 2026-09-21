"""The fork's own documentation, held to the code.

A README that claims a feature the code does not have is worse than one that
claims nothing, and a fork inventory that drifts stops being an inventory.
Both are checked here the same way the MQTT schema is: against what actually
exists.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
FORK = (ROOT / "docs" / "fork-changes.md").read_text()


class TestTheInventoryNamesWhatExists:
    def test_every_package_it_lists_is_there(self):
        for package in ("cluster", "planner", "profiles", "placement",
                        "safety", "measure", "assist", "telemetry",
                        "clients"):
            assert f"ainode/{package}/" in FORK or f"ainode/{package}" in FORK, package
            assert (ROOT / "ainode" / package).is_dir(), package

    def test_every_module_it_points_at_exists(self):
        import re

        for match in re.finditer(r"`(ainode/[\w/]+\.py)`", FORK):
            assert (ROOT / match.group(1)).is_file(), match.group(1)

    def test_the_documents_it_links_exist(self):
        import re

        for match in re.finditer(r"\]\(((?:\.\./)?[\w/.-]+\.md)\)", FORK):
            target = (ROOT / "docs" / match.group(1)).resolve()
            assert target.is_file(), match.group(1)

    def test_it_says_what_it_was_measured_against(self):
        # An inventory without a baseline is a list of opinions.
        assert "upstream/main" in FORK

    def test_it_says_what_was_not_changed(self):
        # Bounding the surface is what makes the rest credible.
        assert "What was not changed" in FORK
        for untouched in ("training", "onboarding", "installer"):
            assert untouched in FORK.lower()

    def test_it_separates_what_belongs_upstream(self):
        assert "worth carrying upstream" in FORK

    def test_it_credits_the_borrowed_work(self):
        assert "eugr/spark-vllm-docker" in FORK
        assert "MIT" in FORK


class TestTheHardwareFactsAreWrittenDown:
    """The things that cost days to learn, kept in the repository rather than
    in someone's head."""

    def test_the_unified_memory_trap(self):
        assert "no separate" in FORK and "VRAM" in FORK
        assert "enable_model_cpu_offload" in FORK

    def test_the_rdma_counter_trap(self):
        assert "32-bit words, not bytes" in FORK
        assert "/proc/net/dev" in FORK

    def test_the_nvidia_smi_reading(self):
        assert "[N/A]" in FORK

    def test_the_tensor_parallel_constraint(self):
        assert "2, 4 or 8" in FORK


class TestTheReadmeMatchesTheCode:
    def test_it_links_the_inventory(self):
        assert "docs/fork-changes.md" in README

    def test_the_fork_section_covers_more_than_the_fabric(self):
        section = README.split("## This is a fork")[1].split("## What AINode is")[0]
        for area in ("planner", "memory guard", "error assistant",
                     "measurement store", "Image generation"):
            assert area in section, area

    def test_it_still_says_who_should_use_upstream(self):
        # The honest part. It was true before and is still true.
        import re

        assert re.search(r"upstream is the better\s+choice", README)

    def test_it_points_contributors_at_the_maintained_project_first(self):
        assert "getainode/ainode](https://github.com/getainode/ainode)" in \
            README.split("## Contributing")[1]

    def test_the_feature_rows_point_at_real_routes(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        # Every feature row below names something the API actually serves.
        assert "/api/planner" in paths
        assert "/api/safety/memory" in paths
        assert "/api/measurements" in paths
        assert "/api/placement" in paths
        assert "/api/assist/diagnose" in paths
        assert "/v1/images/generations" in paths

    def test_the_documents_it_links_exist(self):
        import re

        for match in re.finditer(r"\]\(((?:docs/)?[\w/.-]+\.md)\)", README):
            assert (ROOT / match.group(1)).is_file(), match.group(1)
