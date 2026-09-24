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


class TestTheInventoryNoticesWhenItGoesStale:
    """Documentation kept current by discipline goes stale; kept current by a
    failing test, it does not. These are the two kinds of drift that matter:
    a whole subsystem arriving undocumented, and a number that quietly stops
    being true."""

    def test_every_package_in_the_tree_is_accounted_for(self):
        # A new ainode/<package>/ is a new subsystem. If it is worth a
        # directory it is worth a line in the inventory — and if it is
        # genuinely upstream's, say so there instead.
        # Packages that exist upstream. Adding to this list is a deliberate
        # act: it says "this one is not ours", which is exactly the claim the
        # inventory is about.
        upstream = {"api", "auth", "bench", "cli", "core", "datasets",
                    "discovery", "embeddings", "engine", "metrics", "models",
                    "onboarding", "secrets", "service", "training", "web"}
        packages = {p.name for p in (ROOT / "ainode").iterdir()
                    if p.is_dir() and (p / "__init__.py").exists()}
        for package in sorted(packages - upstream):
            assert package in FORK, (
                f"ainode/{package}/ exists and docs/fork-changes.md does not "
                f"mention it")

    def test_the_test_count_it_claims_is_still_roughly_true(self):
        import re

        claimed = re.search(r"([\d,]+) tests\.", FORK)
        assert claimed, "the inventory should say how large the suite is"
        number = int(claimed.group(1).replace(",", ""))
        functions = sum(
            len(re.findall(r"^\s*def test_", path.read_text(), re.M))
            for path in (ROOT / "tests").glob("test_*.py"))
        # Loose on purpose. The figure in the document is what pytest reports,
        # which counts parametrised cases separately, while this counts the
        # functions that produce them — so the two legitimately differ by a
        # few hundred. The drift this is here to catch is the other kind: a
        # document that still claims a suite half this size.
        assert 0.6 <= number / max(1, functions) <= 1.6, (
            f"docs/fork-changes.md claims {number} tests; there are "
            f"{functions} test functions. Re-run the suite and update it.")


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


class TestTheDocumentedPortIsTheOneWeListenOn:
    """AINode binds web_port (3000). Port 8000 is the ENGINE's own listener —
    a vLLM serving one model, with no federation, no kind filter and no
    /v1/images/generations. The README sent readers there, which works by
    accident for chat on a node that happens to be serving and not at all for
    anything this fork added.

        was muss ich bei openwebui eintragen um auf die bildgenerierung
        zuzugreifen?

    The answer depends on this being right.
    """

    README = (Path(__file__).resolve().parent.parent / "README.md").read_text()

    def test_the_app_binds_the_web_port(self):
        import inspect

        from ainode.api import server

        source = inspect.getsource(server)
        assert "web.run_app(app, host=config.host, port=config.web_port" in source

    def test_no_example_sends_an_ainode_route_to_8000(self):
        for line in self.README.splitlines():
            if ":8000" not in line:
                continue
            # /api/cluster/update-all is a head-to-head call on the API port
            # in a cluster where that is how it is deployed; everything else
            # named here is served by the aiohttp app.
            assert "/v1/" not in line and "/api/metrics" not in line \
                and "/metrics" not in line, line

    def test_the_openai_example_uses_3000(self):
        assert 'base_url="http://localhost:3000/v1"' in self.README

    def test_the_image_recipe_is_there(self):
        assert "/v1/images/generations" in self.README
        assert "Open WebUI" in self.README
        assert "b64_json" in self.README


class TestTheCodingShortlistStaysHonest:
    """A model shortlist is a document that rots faster than code.

        mach mir mal eine liste der top 10 lokalen llms welche auf unserem
        setup so laufen könnten

    The list itself is a judgement and cannot be tested. What can be tested is
    that it still points at real things: modules that exist, constants that
    still hold the values it quotes, and a README that still links it. When
    the planner's reserves change, the budget table in that document is wrong
    and this fails — which is the point.
    """

    SHORTLIST = (Path(__file__).resolve().parent.parent
                 / "docs" / "coding-models.md").read_text()
    README = (Path(__file__).resolve().parent.parent / "README.md").read_text()

    def test_the_readme_points_at_it(self):
        assert "docs/coding-models.md" in self.README

    def test_every_module_it_names_exists(self):
        import re

        for match in re.finditer(r"`(ainode/[\w/]+\.py)", self.SHORTLIST):
            assert (ROOT / match.group(1)).is_file(), match.group(1)

    def test_the_reserve_it_quotes_is_the_reserve_we_keep(self):
        from ainode.planner.compute import SYSTEM_RESERVE_GB

        assert f"| system reserve | {SYSTEM_RESERVE_GB:.0f} |" in self.SHORTLIST

    def test_the_headroom_share_it_quotes_is_the_one_we_apply(self):
        from ainode.planner.compute import PLAN_HEADROOM_SHARE

        assert f"{PLAN_HEADROOM_SHARE * 100:.0f} % of total" in self.SHORTLIST

    def test_it_names_the_expert_parallel_flag_we_actually_emit(self):
        from ainode.models.architecture import EXPERT_PARALLEL

        assert EXPERT_PARALLEL in self.SHORTLIST

    def test_the_kv_arithmetic_is_attributed_to_the_planner(self):
        # Every KV figure in the document comes out of this function; if it is
        # renamed, the figures lose their provenance.
        from ainode.planner import compute

        assert hasattr(compute, "kv_bytes_per_token")
        assert "kv_bytes_per_token" in self.SHORTLIST

    def test_it_says_nothing_here_is_verified_on_the_hardware(self):
        # The one claim that must never quietly disappear.
        assert "has not yet said so" in self.SHORTLIST

    def test_it_gives_a_hardware_check(self):
        assert "/api/planner/plan" in self.SHORTLIST
        assert '"fits": true' in self.SHORTLIST
