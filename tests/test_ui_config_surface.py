"""What an operator can configure without reaching for curl.

Three models served to three groups of people is the target deployment, and
every knob that shapes it — concurrency, context, KV precision, API aliases —
has to be reachable from the launch panel. The Security section matters for the
same reason: auth is off by default, so a shared cluster is open until someone
turns it on, and there was no UI to do that with.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
INDEX = (WEB / "templates" / "index.html").read_text()
APP_JS = (WEB / "static" / "js" / "app.js").read_text()
STYLE = (WEB / "static" / "css" / "style.css").read_text()


class TestLaunchPanelFields:
    @pytest.mark.parametrize(
        "element_id",
        [
            "launch-model", "launch-gmu",
            "launch-max-seqs",        # --max-num-seqs: concurrent requests
            "launch-max-len",         # max_model_len
            "launch-kv-dtype",        # fp8 vs auto — vision models need auto
            "launch-served-name",     # API alias, what OpenWebUI addresses
            "launch-quantization",
            "launch-engine-image",    # Qwen3.8 needs vllm/vllm-openai:v0.27.1
            "launch-extra-args",
            "launch-trust-remote-code",
        ],
    )
    def test_field_exists(self, element_id):
        assert f'id="{element_id}"' in INDEX

    def test_every_field_is_read_by_the_launcher(self):
        for element_id in ("launch-max-seqs", "launch-max-len", "launch-kv-dtype",
                           "launch-served-name", "launch-quantization",
                           "launch-engine-image", "launch-extra-args",
                           "launch-trust-remote-code"):
            assert f"'{element_id}'" in APP_JS, f"{element_id} is rendered but never read"

    def test_sent_field_names_match_the_api(self):
        """A field the API does not know is silently ignored, which looks like
        the setting had no effect."""
        from ainode.models.api_routes import _OVERRIDE_KEYS

        for key in ("kv_cache_dtype", "quantization", "engine_image",
                    "served_model_name", "trust_remote_code", "extra_vllm_args",
                    "max_model_len"):
            assert f"advanced.{key}" in APP_JS or f'"{key}"' in APP_JS
        # All but max_model_len/trust_remote_code are in the persisted set.
        assert "kv_cache_dtype" in _OVERRIDE_KEYS
        assert "extra_vllm_args" in _OVERRIDE_KEYS

    def test_kv_dtype_offers_auto_for_vision_models(self):
        """fp8 KV corrupts vision models on GB10 — the option has to exist."""
        assert 'value="auto"' in INDEX
        assert "vision" in INDEX

    def test_advanced_is_collapsed_by_default(self):
        """Nine extra fields must not push LAUNCH off the panel."""
        assert '<details class="launch-advanced"' in INDEX
        assert '<details class="launch-advanced" id="launch-advanced" open' not in INDEX


class TestSecuritySection:
    def test_nav_entry_exists(self):
        assert 'data-section="security"' in INDEX

    def test_renderer_is_wired(self):
        assert "case 'security'" in APP_JS
        assert "renderConfigSecurity()" in APP_JS

    def test_it_says_auth_is_off_by_default(self):
        """The single most important fact about a shared cluster."""
        assert "off by default" in APP_JS

    def test_key_is_shown_once_in_a_blocking_panel(self):
        assert "_showOneTimeKey" in APP_JS
        assert "onetime-key-backdrop" in APP_JS
        assert ".onetime-key-backdrop" in STYLE

    def test_copy_falls_back_without_a_secure_context(self):
        """A LAN node on plain http has no navigator.clipboard."""
        assert "navigator.clipboard" in APP_JS
        assert "Ctrl+C" in APP_JS

    def test_revocation_is_offered_per_key(self):
        assert "data-revoke" in APP_JS
        assert "/api/auth/keys/" in APP_JS


class TestAuthApiSupportsTheUi:
    def _status(self, cfg):
        from ainode.auth.api_routes import handle_auth_status

        class _Req:
            app = {"auth_config": cfg}

        return json.loads(asyncio.run(handle_auth_status(_Req())).body)

    def test_status_exposes_ids_so_keys_can_be_revoked(self, tmp_path, monkeypatch):
        from ainode.auth.middleware import AuthConfig

        monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
        cfg = AuthConfig()
        cfg.generate_key(name="anna")
        cfg.generate_key(name="bob")
        body = self._status(cfg)
        assert body["key_count"] == 2
        assert {k["name"] for k in body["keys"]} == {"anna", "bob"}
        assert all(k["id"] for k in body["keys"])

    def test_status_never_leaks_the_key_or_its_hash(self, tmp_path, monkeypatch):
        from ainode.auth.middleware import AuthConfig

        monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
        cfg = AuthConfig()
        created = cfg.generate_key(name="anna")
        body = self._status(cfg)
        blob = json.dumps(body)
        assert created["key"] not in blob
        assert "key_hash" not in blob

    def test_keys_carry_a_creation_time(self, tmp_path, monkeypatch):
        from ainode.auth.middleware import AuthConfig

        monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
        cfg = AuthConfig()
        cfg.generate_key()
        assert self._status(cfg)["keys"][0]["created_at"] > 0


class TestEmbeddingsSurviveARestart:
    """They are in-process, so a restart drops them and a RAG pipeline breaks
    silently. LLM instances were already replayed; these were not."""

    def _manager(self, tmp_path, monkeypatch):
        from ainode.embeddings.manager import EmbeddingManager

        monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
        return EmbeddingManager()

    def test_manifest_round_trips(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["nomic-ai/nomic-embed-text-v1.5"] = object()
        m.save_manifest()
        assert m.load_manifest() == ["nomic-ai/nomic-embed-text-v1.5"]

    def test_missing_manifest_is_empty_not_an_error(self, tmp_path, monkeypatch):
        assert self._manager(tmp_path, monkeypatch).load_manifest() == []

    def test_unload_is_recorded_so_it_stays_unloaded(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["a"] = object()
        m._models["b"] = object()
        m.save_manifest()
        m._models.pop("a")
        m.save_manifest()
        assert m.load_manifest() == ["b"]

    def test_replay_skips_what_is_already_loaded(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["a"] = object()
        m.save_manifest()
        calls = []
        monkeypatch.setattr(type(m), "load", lambda self, mid, force=False: calls.append(mid))
        m.replay()
        assert calls == []

    def test_one_broken_model_does_not_block_the_others(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["broken"] = object()
        m._models["good"] = object()
        m.save_manifest()
        m._models.clear()
        loaded = []

        def _load(self, mid, force=False):
            if mid == "broken":
                raise RuntimeError("gone from the hub")
            loaded.append(mid)

        monkeypatch.setattr(type(m), "load", _load)
        m.replay()          # must not raise
        assert loaded == ["good"]


class TestLauncherIsInTheImage:
    """EugrBackend shells out to /opt/spark-vllm-docker/launch-cluster.sh. It
    was present when this image was FROM ainode-base; since the split it is
    not, and every launch failed with "eugr launcher missing"."""

    DOCKERFILE = (Path(__file__).resolve().parent.parent
                  / "scripts" / "Dockerfile.ainode").read_text()
    BUILD_BASE = (Path(__file__).resolve().parent.parent
                  / "scripts" / "build-base-image.sh").read_text()

    def test_the_dockerfile_fetches_it(self):
        assert "/opt/spark-vllm-docker" in self.DOCKERFILE
        assert "launch-cluster.sh" in self.DOCKERFILE

    def test_the_fetch_is_verified_not_assumed(self):
        """A silent 404 would leave the same missing-launcher failure."""
        assert "test -x /opt/spark-vllm-docker/launch-cluster.sh" in self.DOCKERFILE

    def test_the_pin_matches_the_engine_build(self):
        """Launcher and engine hand each other a .env and a launch script, so a
        drifted pin is a wrong-shape contract rather than a clean failure."""
        import re

        dockerfile_sha = re.search(r"ARG EUGR_COMMIT=([0-9a-f]{40})", self.DOCKERFILE)
        base_sha = re.search(r'EUGR_COMMIT="\$\{EUGR_COMMIT:-([0-9a-f]{40})\}"',
                             self.BUILD_BASE)
        assert dockerfile_sha and base_sha
        assert dockerfile_sha.group(1) == base_sha.group(1)


class TestLoadPhaseIsReportedByBothBackends:
    """The UI turns the phase into a percentage. The eugr backend — the default
    — reported none, so every launch showed a flat 8% (the fallback for
    "unknown") for the whole of a load that can take minutes. That is
    indistinguishable from a hang."""

    def _tracker(self):
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        return t

    def test_phases_advance_in_order(self):
        t = self._tracker()
        assert t.phase == "starting"
        t.observe("Loading model weights took 12.3 GiB")
        assert t.phase == "loading_weights"
        t.observe("NCCL INFO Bootstrap : Using enP7s7")
        assert t.phase == "distributed_init"
        t.observe("Memory profiling results: total_gpu_memory=...")
        assert t.phase == "profiling"

    def test_a_phase_never_goes_backwards(self):
        t = self._tracker()
        t.observe("Memory profiling results")
        t.observe("Loading model weights")      # a late straggler line
        assert t.phase == "profiling"

    def test_readiness_fires_once(self):
        t = self._tracker()
        assert t.observe("INFO:     Uvicorn running on http://0.0.0.0:8000") is True
        assert t.observe("INFO:     Application startup complete.") is False
        assert t.phase == "ready"

    def test_launcher_progress_counts_too(self):
        """launch-cluster.sh talks before vLLM does; without these the bar sits
        at 'starting' through the slowest part of a distributed launch."""
        t = self._tracker()
        t.observe("Waiting for cluster to be ready...")
        assert t.phase == "distributed_init"

    def test_unknown_lines_do_not_throw(self):
        """vLLM's log format is not a contract."""
        t = self._tracker()
        for line in ("", "\n", "🚀 emoji", "a" * 5000):
            t.observe(line)
        assert t.phase == "starting"

    def test_the_api_poll_latch_wins(self):
        """wait_ready() can beat the log stream; the phase must not stay stuck
        on a model that is already serving."""
        t = self._tracker()
        assert t.current(ready_latch=True) == "ready"

    def test_eugr_exposes_load_phase(self):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends.eugr import EugrBackend

        backend = EugrBackend(NodeConfig(node_id="n"))
        assert backend.load_phase == "idle"

    def test_both_backends_share_one_phase_vocabulary(self):
        from ainode.engine.backends.nvidia import NvidiaBackend
        from ainode.engine.load_phase import LOAD_PHASE_ORDER

        assert NvidiaBackend._LOAD_PHASE_ORDER is LOAD_PHASE_ORDER

    def test_every_phase_the_ui_knows_is_produced(self):
        """A phase the UI has no entry for falls back to 8% — the bug this
        fixes, in a different disguise."""
        ui = {"idle", "starting", "distributing", "loading_weights",
              "distributed_init", "profiling", "ready"}
        from ainode.engine.load_phase import LOAD_PHASE_ORDER

        assert set(LOAD_PHASE_ORDER) - ui == set()

    def test_the_ui_table_lists_every_phase(self):
        """A phase with no entry renders as the idle fallback — a flat 8% that
        reads as a hang. That is the bug this whole class exists for."""
        from ainode.engine.load_phase import LOAD_PHASE_ORDER

        for phase in LOAD_PHASE_ORDER:
            assert f"{phase}:" in APP_JS, f"PHASE_INFO has no entry for {phase}"


class TestADeadLaunchIsReportedAsDead:
    """A launcher that dies looked exactly like a model that takes minutes to
    load: the card sat at a phase it would never leave, and the only way to
    find out was to go and read a log file."""

    def _tracker(self):
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        return t

    def test_failure_is_terminal_and_carries_a_reason(self):
        t = self._tracker()
        t.observe("Error: Passwordless SSH to 192.168.1.3 failed.")
        t.fail("the launcher exited (code 1)")
        assert t.phase == "failed"
        assert "code 1" in t.failure_reason()
        assert "Passwordless SSH" in t.failure_reason()

    def test_a_running_engine_is_never_retracted(self):
        """The launcher exiting after a successful start is normal for a
        detached engine and must not mark a serving model failed."""
        t = self._tracker()
        t.observe("INFO:     Uvicorn running on http://0.0.0.0:8000")
        t.fail("the launcher exited (code 0)")
        assert t.phase == "ready"
        assert t.failure_reason() == ""

    def test_no_reason_before_a_failure(self):
        assert self._tracker().failure_reason() == ""

    def test_the_tail_is_bounded(self):
        """A launcher that fails after thousands of lines must not carry them
        all into a status payload polled every few seconds."""
        t = self._tracker()
        for i in range(5000):
            t.observe(f"line {i}")
        t.fail("died")
        assert len(t.tail) <= 12
        assert len(t.failure_reason()) < 500

    def test_blank_lines_are_not_quoted_back(self):
        t = self._tracker()
        t.observe("real failure line")
        t.observe("   \n")
        t.fail("died")
        assert "real failure line" in t.failure_reason()

    def test_reset_clears_a_previous_failure(self):
        """A relaunch must not inherit the last one's error."""
        t = self._tracker()
        t.fail("old failure")
        t.reset()
        assert t.phase == "starting"
        assert t.failure_reason() == ""

    def test_both_backends_expose_the_error(self):
        from ainode.core.config import NodeConfig
        from ainode.engine.backends.eugr import EugrBackend
        from ainode.engine.backends.nvidia import NvidiaBackend

        for cls in (EugrBackend, NvidiaBackend):
            backend = cls(NodeConfig(node_id="n"))
            assert backend.load_error == ""
            backend._phase.reset()
            backend._phase.fail("boom")
            assert "boom" in backend.load_error

    def test_the_ui_renders_failed_terminally(self):
        assert "failed: ['failed', 100]" in APP_JS
        assert "instance-status failed" in APP_JS
        assert "instance-failed-note" in APP_JS
        assert ".instance-failed-note" in STYLE

    def test_the_api_exposes_the_error(self):
        from ainode.api import server

        assert '"load_error"' in (server.__file__ and
                                  Path(server.__file__).read_text())


class TestRootCauseBeatsTheTail:
    """vLLM reports an engine crash twice: the real exception in the worker,
    then "Engine core initialization failed. See root cause above" from the
    supervisor — and that useless second one is what the tail of the log
    actually contains. Quoting the tail told the operator nothing."""

    # Trimmed from a real failure: a DFlash draft model served as a base.
    REAL_LOG = [
        "(EngineCore pid=143)   File \"/usr/local/lib/python3.12/dist-packages/"
        "vllm/model_executor/models/qwen3_dflash.py\", line 693, in __init__",
        "(EngineCore pid=143)     self.draft_model_config = "
        "vllm_config.speculative_config.draft_model_config",
        "(EngineCore pid=143) AttributeError: 'NoneType' object has no attribute "
        "'draft_model_config'",
        "[rank0]:[W912 00:26:00.238913763 ProcessGroupNCCL.cpp:1575] Warning: ...",
        "(APIServer pid=91) Traceback (most recent call last):",
        "(APIServer pid=91)   File \"/usr/local/bin/vllm\", line 10, in <module>",
        "(APIServer pid=91)     sys.exit(main())",
        "(APIServer pid=91)   File \".../utils.py\", line 1320, in wait_for_engine_startup",
        "(APIServer pid=91)     raise RuntimeError(",
        "(APIServer pid=91) RuntimeError: Engine core initialization failed. "
        "See root cause above. Failed core proc(s): {}",
    ]

    def _fail_with(self, lines):
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        for line in lines:
            t.observe(line)
        t.fail("the launcher exited (code 1)")
        return t

    def test_the_real_exception_is_reported(self):
        """A crash with no recognised cause falls back to its own exception —
        the drafter case has a named hint that outranks it, tested separately."""
        reason = self._fail_with([
            "(EngineCore pid=143)   File \".../model_runner.py\", line 400",
            "(EngineCore pid=143) torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 4.00 GiB",
            "(APIServer pid=91) RuntimeError: Engine core initialization failed. "
            "See root cause above.",
        ]).failure_reason()
        assert "OutOfMemoryError" in reason
        assert "4.00 GiB" in reason

    def test_the_useless_supervisor_error_is_not(self):
        reason = self._fail_with(self.REAL_LOG).failure_reason()
        assert "See root cause above" not in reason

    def test_the_root_cause_is_still_captured_under_a_hint(self):
        """The hint is what gets shown, but the exception is not discarded —
        a future caller may want both."""
        t = self._fail_with(self.REAL_LOG)
        assert "draft_model_config" in t.root_cause

    def test_the_process_prefix_is_stripped(self):
        t = self._fail_with(self.REAL_LOG)
        assert "EngineCore pid=" not in t.root_cause

    def test_the_first_exception_wins(self):
        t = self._fail_with([
            "ValueError: the real problem",
            "RuntimeError: a downstream consequence",
        ])
        assert "ValueError: the real problem" in t.failure_reason()
        assert "downstream" not in t.failure_reason()

    def test_it_falls_back_to_the_tail_without_an_exception_line(self):
        """Not every failure is a Python traceback — a launcher can die on an
        SSH error with no exception anywhere."""
        t = self._fail_with([
            "Checking SSH connectivity to worker nodes...",
            "Error: Passwordless SSH to 192.168.1.3 failed.",
        ])
        assert "Passwordless SSH" in t.failure_reason()

    @pytest.mark.parametrize("line", [
        "AttributeError: x",
        "(EngineCore pid=1) ValueError: x",
        "[rank0] RuntimeError: x",
        "torch.cuda.OutOfMemoryError: CUDA out of memory",
    ])
    def test_exception_shapes_recognised(self, line):
        assert self._fail_with([line]).root_cause

    @pytest.mark.parametrize("line", [
        "INFO 09-12 00:26:00 [utils.py:620] starting",
        "  File \"/usr/local/bin/vllm\", line 10, in <module>",
        "Traceback (most recent call last):",
        "just some prose about an Error: that is not one",
    ])
    def test_non_exceptions_are_not_mistaken_for_one(self, line):
        assert not self._fail_with([line]).root_cause

    def test_reset_clears_the_root_cause(self):
        t = self._fail_with(["ValueError: old"])
        t.reset()
        assert t.root_cause == ""


class TestKnownMistakesAreNamed:
    """Some failures have a traceback that describes the symptom and not the
    mistake. A DFlash repository is the case that prompted this: vLLM dies with
    "AttributeError: 'NoneType' object has no attribute 'draft_model_config'",
    which says nothing about what the operator actually did."""

    def _fail_with(self, lines):
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        for line in lines:
            t.observe(line)
        t.fail("the launcher exited (code 1)")
        return t

    REAL = [
        "(APIServer pid=91) INFO 09-12 00:45:55 [model.py:692] "
        "Resolved architecture: DFlashDraftModel",
        "(EngineCore pid=143) AttributeError: 'NoneType' object has no attribute "
        "'draft_model_config'",
        "(APIServer pid=91) RuntimeError: Engine core initialization failed. "
        "See root cause above.",
    ]

    def test_a_draft_model_is_named_as_such(self):
        reason = self._fail_with(self.REAL).failure_reason()
        assert "DRAFT model" in reason
        assert "--speculative-config" in reason

    def test_it_outranks_the_raw_traceback(self):
        """The AttributeError is accurate and useless; the hint is neither."""
        reason = self._fail_with(self.REAL).failure_reason()
        assert "draft_model_config" not in reason

    @pytest.mark.parametrize("arch", [
        "Resolved architecture: DFlashDraftModel",
        "Resolved architecture: Qwen3DraftModel",
        "resolved architecture: SomeDraftModel",
    ])
    def test_draft_architectures_recognised(self, arch):
        assert self._fail_with([arch]).fatal_hint

    @pytest.mark.parametrize("arch", [
        "Resolved architecture: Gemma3ForCausalLM",
        "Resolved architecture: Qwen3MoeForCausalLM",
        "some prose mentioning a draft model in passing",
    ])
    def test_ordinary_models_are_not_flagged(self, arch):
        assert not self._fail_with([arch]).fatal_hint

    def test_reset_clears_the_hint(self):
        t = self._fail_with(self.REAL)
        t.reset()
        assert t.fatal_hint == ""

    def test_a_successful_load_is_unaffected(self):
        """The hint must not fire on a model that goes on to serve."""
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        t.observe("Resolved architecture: DFlashDraftModel")
        t.observe("INFO:     Uvicorn running on http://0.0.0.0:8000")
        t.fail("the launcher exited (code 0)")
        assert t.phase == "ready"
        assert t.failure_reason() == ""


class TestDrafterIsRefusedBeforeLaunching:
    """Catching it in the log costs two minutes of loading first. The catalog
    already knows which repos are drafters — a curated entry that pairs with
    one names it in its speculative-config flags."""

    def test_a_known_drafter_names_its_base(self):
        from ainode.models.api_routes import drafter_base_model

        assert drafter_base_model(
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
        ) == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"

    def test_the_base_is_not_mistaken_for_its_own_drafter(self):
        """A base repo id is a PREFIX of its drafter's, so a substring test
        flags the correct model and sends the operator in a circle."""
        from ainode.models.api_routes import drafter_base_model

        assert drafter_base_model(
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
        ) == ""

    @pytest.mark.parametrize("model", [
        "QuantTrio/Qwen3.5-4B-AWQ", "unsloth/Qwen3.8-27B-NVFP4",
        "google/gemma-4-26B-A4B-it", "", "   ",
    ])
    def test_ordinary_models_pass(self, model):
        from ainode.models.api_routes import drafter_base_model

        assert drafter_base_model(model) == ""

    def test_json_style_speculative_config_is_understood(self):
        """Two spellings are in use: a flat --speculative_config.model pair and
        a --speculative-config JSON blob."""
        from ainode.models.api_routes import _names_a_drafter

        assert _names_a_drafter(
            ["--speculative-config", '{"method":"dflash","model":"org/drafter"}'],
            "org/drafter",
        )
        assert _names_a_drafter(
            ["--speculative_config.model", "org/drafter"], "org/drafter"
        )
        assert not _names_a_drafter(
            ["--speculative-config", '{"model":"org/other"}'], "org/drafter"
        )

    def test_malformed_json_does_not_throw(self):
        from ainode.models.api_routes import _names_a_drafter

        assert not _names_a_drafter(["--speculative-config", "{not json"], "x")

    def test_both_launch_routes_refuse_it(self):
        """/api/models/load and /api/sharding/launch are separate entry points
        and an operator can reach either from the UI."""
        from ainode.models import api_routes as m
        from ainode.engine import sharding_routes as sh

        assert "drafter_base_model" in Path(m.__file__).read_text()
        assert "drafter_base_model" in Path(sh.__file__).read_text()
        assert "load_instead" in Path(m.__file__).read_text()

    @pytest.mark.parametrize("arch", [
        "DFlashDraftModel",      # z-lab gemma DFlash
        "Qwen3DSparkModel",      # nvidia Nemotron DSpark
        "SomeEagle3Model",
    ])
    def test_the_log_net_covers_architectures_that_share_no_suffix(self, arch):
        """The two real cases here spelled it differently, and the next vendor
        will too."""
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        t.observe(f"INFO [model.py:692] Resolved architecture: {arch}")
        assert t.fatal_hint

    def test_the_log_net_catches_an_unenumerated_drafter(self):
        """Whatever the architecture is called, it dies reaching through a
        speculative_config that is None."""
        from ainode.engine.load_phase import LoadPhaseTracker

        t = LoadPhaseTracker()
        t.reset()
        t.observe("(EngineCore pid=1) AttributeError: 'NoneType' object has no "
                  "attribute 'draft_model_config'")
        assert t.fatal_hint
