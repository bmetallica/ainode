"""GLM 5.3 Flash runs on an engine image nothing else uses.

Its flags — --attention-backend B12X, --moe-backend b12x, --load-format b12x —
exist only in the experimental B12X build, and its configuration is half
environment variables with no command-line equivalent at all. Both of those
had no route through AINode: the catalog could name one image, which the eugr
path declines because it names the NVIDIA path's, and the launch panel had no
field for engine environment.

Untested on hardware. Everything here is upstream's recipe carried over; these
tests check that it arrives at the engine intact, not that the engine likes it.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend
from ainode.engine.parallelism import ParallelPlan, Strategy, plan_for_model
from ainode.models.api_routes import (
    apply_catalog_recipe,
    catalog_proven_tp,
    catalog_recipe,
)
from ainode.models.registry import CURATED_CLUSTER_MODELS

MODEL = "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
INDEX = (WEB / "templates" / "index.html").read_text()
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class TestCatalogEntry:
    def test_it_names_its_own_engine_image(self):
        assert catalog_recipe(MODEL)["engine_image_eugr"] == "vllm-node-b12x"

    def test_it_does_not_pin_the_nvidia_image(self):
        # Nobody has run this on that lineage; claiming otherwise would send a
        # 20 GB pull after an image that cannot run it.
        assert "engine_image" not in catalog_recipe(MODEL)

    def test_it_is_proven_at_two_nodes(self):
        # cluster_only in the upstream recipe: it does not run on one.
        assert catalog_proven_tp(MODEL) == 2

    def test_it_is_marked_verified(self):
        # Served on the cluster on 2026-09-14 at TP=2, 1,072,101 tokens of KV
        # cache. It was unverified for as long as that was untrue.
        assert CURATED_CLUSTER_MODELS["glm-5.3-flash-nvfp4-spark"].verified is True

    def test_the_b12x_flags_are_all_there(self):
        args = catalog_recipe(MODEL)["extra_vllm_args"]
        for flag in ("--attention-backend", "--moe-backend", "--linear-backend",
                     "--load-format", "--quantization", "--kv-cache-memory-bytes"):
            assert flag in args, flag
        assert "b12x" in args and "B12X" in args

    def test_the_environment_comes_with_it(self):
        env = catalog_recipe(MODEL)["extra_env"]
        assert env["CUTE_DSL_ARCH"] == "sm_121a"
        assert env["B12X_POLICY_MODE"] == "auto"
        assert len([k for k in env if k.startswith("INSTANTTENSOR_")]) == 5

    def test_its_size_is_not_invented(self):
        # Not published anywhere we can check. min_memory_gb states the real
        # constraint (more than one Spark); size_gb stays 0 rather than
        # feeding the fit calculator a number we made up.
        info = CURATED_CLUSTER_MODELS["glm-5.3-flash-nvfp4-spark"]
        assert info.size_gb == 0.0
        assert info.min_memory_gb > 122


class TestItReachesTheEngine:
    def _backend(self):
        overrides, gmu = apply_catalog_recipe(MODEL, {}, None)
        backend = EugrBackend.__new__(EugrBackend)
        backend.config = NodeConfig(model=MODEL, models_dir=tempfile.mkdtemp(),
                                    gpu_memory_utilization=gmu, **overrides)
        return backend

    def test_the_launcher_is_told_which_image(self):
        assert self._backend()._launcher_image_args() == ["-t", "vllm-node-b12x"]

    def test_a_catalog_image_for_the_other_backend_is_still_declined(self):
        # The rule this was built on top of: Qwen3.8 names the NVIDIA image,
        # which the launcher cannot use.
        overrides, _ = apply_catalog_recipe("unsloth/Qwen3.8-27B-NVFP4", {}, None)
        backend = EugrBackend.__new__(EugrBackend)
        backend.config = NodeConfig(model="unsloth/Qwen3.8-27B-NVFP4",
                                    models_dir=tempfile.mkdtemp(), **overrides)
        assert backend._launcher_image_args() == []

    def test_two_nodes_plan_as_tensor_parallel(self):
        plan, note = plan_for_model(Strategy.AUTO, 2, catalog_proven_tp(MODEL))
        assert plan.tensor_parallel_size == 2 and note == ""

    def test_the_b12x_values_survive_the_shell(self):
        backend = self._backend()
        script = Path(backend._write_launch_script(ParallelPlan(tensor_parallel_size=2)))
        bindir = Path(tempfile.mkdtemp())
        out = bindir / "argv.json"
        stub = bindir / "vllm"
        stub.write_text("#!/usr/bin/env python3\nimport json,sys\n"
                        f"open({str(out)!r},'w').write(json.dumps(sys.argv[1:]))\n")
        stub.chmod(0o755)
        subprocess.run(["bash", str(script)],
                       env=dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}"),
                       capture_output=True, timeout=60)
        argv = json.loads(out.read_text())
        # The speculative config is gone from this recipe (see
        # TestWhatTheHardwareContradicted); the attention backend is the
        # remaining B12X argument whose value must survive the shell.
        assert argv[argv.index("--attention-backend") + 1] == "B12X"
        assert argv[argv.index("--kv-cache-memory-bytes") + 1] == "8G"

    def test_the_recipe_dtype_is_not_duplicated(self):
        # The writer adds --dtype bfloat16 on unified memory; the recipe sets
        # it too, and vLLM should not be handed the flag twice.
        backend = self._backend()
        script = Path(backend._write_launch_script(ParallelPlan())).read_text()
        assert script.count("--dtype") == 1


class TestEngineEnvironmentInTheUi:
    def test_the_field_exists(self):
        assert 'id="launch-extra-env"' in INDEX

    def test_it_is_parsed_into_an_object(self):
        assert "advanced.extra_env" in APP_JS

    def test_comments_and_blank_lines_are_tolerated(self):
        # It is a textarea people paste into; a stray blank line must not
        # produce an empty variable name.
        assert "charAt(0) === '#'" in APP_JS
        assert "eq <= 0" in APP_JS


class TestTheImageScript:
    SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build-b12x-image.sh"

    def test_it_exists_and_is_executable(self):
        assert self.SCRIPT.stat().st_mode & 0o111

    def test_shell_syntax(self):
        assert subprocess.run(["bash", "-n", str(self.SCRIPT)]).returncode == 0

    def test_it_pins_the_image_commit_separately(self):
        # --exp-b12x does not exist at the commit the base image is pinned to;
        # it arrived upstream later. Reusing that pin produced upstream's usage
        # text and nothing else. The launcher (from our own image) and the
        # engine image only have to agree about the .env contract, not about
        # which kernels were compiled in.
        text = self.SCRIPT.read_text()
        assert "EUGR_B12X_COMMIT" in text
        assert "--exp-b12x" in text

    def test_it_checks_the_flag_exists_before_using_it(self):
        # Otherwise the failure is upstream's usage text, which says nothing
        # about pins.
        text = self.SCRIPT.read_text()
        assert 'grep -q -- "--exp-b12x"' in text
        assert "has no --exp-b12x" in text

    def test_it_does_not_share_the_base_image_checkout(self):
        # build-base-image.sh patches its tree; a checkout of another commit
        # there fails, or succeeds and leaves that build on the wrong source.
        text = self.SCRIPT.read_text()
        assert "_eugr-b12x" in text
        assert 'WORKTREE="$SCRIPT_DIR/_eugr"' not in text


class TestTheContextLengthMatchesTheCheckpoint:
    """The recipe asks for 1M; the checkpoint says 256K, and vLLM refuses.

    From the cluster:

        ValidationError: User-specified max_model_len (1048576) is greater
        than the derived max_model_len (max_position_embeddings=262144.0 in
        model's config.json)

    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 would silence it, and vLLM's own warning
    says why not: positions beyond the derived maximum produce NaN under RoPE.
    """

    def test_the_launch_asks_for_what_the_model_has(self):
        args = catalog_recipe(MODEL)["extra_vllm_args"]
        assert args[args.index("--max-model-len") + 1] == "262144"

    def test_the_catalog_advertises_the_same(self):
        assert CURATED_CLUSTER_MODELS["glm-5.3-flash-nvfp4-spark"].context_length == 262144

    def test_the_reason_is_the_measured_one(self):
        # The 262144 limit originally came from the MTP module's config, and
        # dropping the speculative config removed that constraint — the main
        # architecture accepts 1M. What remains is that 1M does not fit: a
        # three-node launch with it was killed for memory during startup.
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
               "registry.py").read_text()
        assert "exit 137" in src
        assert "no longer a hard limit" in src

    def test_the_override_is_not_used(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
               "registry.py").read_text()
        assert "VLLM_ALLOW_LONG_MAX_MODEL_LEN" not in _env_values(src)

    def test_the_deviation_is_recorded(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
               "registry.py").read_text()
        assert "derived max_model_len (262144)" in src


def _env_values(src: str) -> str:
    """The extra_env blocks only — the comment may name the variable."""
    out = []
    for start in range(len(src)):
        if src.startswith("extra_env={", start):
            out.append(src[start:src.index("}", start)])
    return "\n".join(out)


class TestTheB12xLoaderFailureIsNamed:
    """Observed on the cluster, with two hundred lines of fallout after it:

        RuntimeError: the initial b12x loader requires GPU host page tables

    Then Ray workers dying and actor handles from a dead session — none of
    which an operator can act on. The first line is the whole story.
    """

    def _failure(self, *lines):
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        for line in lines:
            tracker.observe(line)
        tracker.fail("the launcher exited (code 1)")
        return tracker.failure_reason()

    def test_it_names_the_loader_and_the_way_out(self):
        reason = self._failure(
            "(EngineCore pid=681) RuntimeError: the initial b12x loader "
            "requires GPU host page tables")
        assert "b12x loader" in reason
        assert "--load-format auto" in reason

    def test_the_ray_fallout_does_not_become_the_explanation(self):
        reason = self._failure(
            "(EngineCore pid=681) RuntimeError: the initial b12x loader "
            "requires GPU host page tables",
            "(EngineCore pid=681) ray.exceptions.ActorHandleNotFoundError: "
            "ActorHandle objects are not valid across Ray sessions",
        )
        assert "ActorHandle" not in reason

    def test_an_ordinary_failure_gets_no_b12x_hint(self):
        # With no exception in the log the reason quotes the tail, which is
        # the existing behaviour; what must not happen is this hint appearing
        # on a failure that has nothing to do with the loader.
        reason = self._failure("INFO [core.py:372] init engine took 150 s")
        assert "--load-format auto" not in reason


class TestWhatTheHardwareContradicted:
    """Three flags from the upstream recipe do not work on this setup.

    Each was measured on the cluster, not reasoned about:

      --max-model-len 1048576   the checkpoint says max_position_embeddings
                                262144, and vLLM refuses the mismatch
      --load-format b12x        RuntimeError: the initial b12x loader requires
                                GPU host page tables
      --speculative-config …    HFValidationError on the local model path;
                                without it the same launch reached warmup
    """

    def test_the_loader_is_the_ordinary_one(self):
        args = catalog_recipe(MODEL)["extra_vllm_args"]
        assert args[args.index("--load-format") + 1] == "auto"

    def test_no_speculative_config(self):
        args = catalog_recipe(MODEL)["extra_vllm_args"]
        assert not any("speculative" in a for a in args)

    def test_the_b12x_backends_are_still_there(self):
        # Only the loader was dropped; the stack the model needs remains.
        args = catalog_recipe(MODEL)["extra_vllm_args"]
        assert args[args.index("--moe-backend") + 1] == "b12x"
        assert args[args.index("--linear-backend") + 1] == "b12x"
        assert args[args.index("--attention-backend") + 1] == "B12X"

    def test_each_deviation_is_recorded_with_its_evidence(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
               "registry.py").read_text()
        for evidence in ("derived max_model_len (262144)",
                         "GPU host page",
                         "HFValidationError"):
            assert evidence in src, evidence

    def test_the_description_says_two_nodes_and_why(self):
        # Two is not a preference, it is the only shape: no SupportsPP, and
        # tensor needs a power-of-two rank count. An operator with three
        # machines will otherwise keep trying to use all three.
        info = CURATED_CLUSTER_MODELS["glm-5.3-flash-nvfp4-spark"]
        assert "exactly two nodes" in info.description
        assert "SupportsPP" in info.description

    def test_the_kv_cap_is_kept_and_both_measurements_are_recorded(self):
        """The 8G cap was briefly dropped on an argument that did not hold.

        1,072,101 tokens was read as "about 9.9 GB, so the cap cannot have
        been in force" — which assumes a constant bytes-per-token. This is a
        hybrid Mamba model launched with --mamba-cache-mode align, where the
        state cache is sized by max_num_seqs; that and max_model_len both
        differed between the two logged launches. The cap was active in the
        good one, so what removing it does is still unmeasured.
        """
        from pathlib import Path

        args = catalog_recipe(MODEL)["extra_vllm_args"]
        assert args[args.index("--kv-cache-memory-bytes") + 1] == "8G"
        src = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
               "registry.py").read_text()
        # Both launches, so the next reader can check the arithmetic.
        assert "434,176 tokens" in src and "1,072,101 tokens" in src


class TestTheReasoningBudgetIsStated:
    """An empty answer with finish_reason=length is not a server fault.

    Measured on the cluster: 979 reasoning tokens for "Zähle von 1 bis 30",
    with zero content tokens when max_tokens was 512 — the whole budget went
    into thinking and the reply never started. A client configured with a
    normal-looking output limit therefore sees the model "stop" mid-session,
    which is exactly how it was reported.

    Note what does NOT fix it: chat_template_kwargs {"enable_thinking": false}
    returns reasoning_tokens=0 but moves the same thinking into `content`.
    It suppresses the separation, not the reasoning.
    """

    def test_the_catalog_warns_about_it(self):
        info = CURATED_CLUSTER_MODELS["glm-5.3-flash-nvfp4-spark"]
        assert "979 reasoning tokens" in info.description
        assert "finish_reason=length" in info.description


class TestTheBlockSizeIsNotOptional:
    """256 is the model's requirement, not a tuning knob.

    Dropping to --block-size 16, to move off the experimental B12X attention
    path while diagnosing a cudaErrorIllegalAddress, failed at startup:

        ValueError: GLM C4 indexing requires a model block size divisible
        by 256

    Recorded so the suggestion is not made again — including by whoever is
    reading the recipe and sees an unusually large block size.
    """

    def test_the_requirement_is_recorded_next_to_the_flag(self):
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "ainode" / "models" /
               "registry.py").read_text()
        assert "GLM C4 indexing requires a model block size" in src

    def test_the_recipe_still_asks_for_256(self):
        args = catalog_recipe(MODEL)["extra_vllm_args"]
        assert args[args.index("--block-size") + 1] == "256"


class TestTheMixedBitHint:
    """A launch that no flag can rescue should say so.

    aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx died with

        Value error, Unsupported weight_bits: 16, currently only support
        {8, 2, 3, 4}

    and the operator's next move was to try different serve flags, which
    cannot work. The 16 is not a mistake in the checkpoint: its
    quantization_config carries bits=16 as the GLOBAL default and 22,249
    per-layer overrides naming the real widths. Stock vLLM reads the default,
    finds 16 and stops; reading the overrides is what the vendor's plugin
    exists to do.
    """

    def _reason(self, line: str) -> str:
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        tracker.observe(line)
        tracker.fail("the launcher exited (code 1)")
        return tracker.failure_reason()

    def test_it_fires_on_the_measured_line(self):
        reason = self._reason(
            "(APIServer pid=551) Value error, Unsupported weight_bits: 16, "
            "currently only support {8, 2, 3, 4}.")
        assert "mixed-bit" in reason or "different width" in reason

    def test_it_says_no_flag_will_help(self):
        """The operator's instinct is to try more arguments. Two hours of that
        is the cost of not saying this."""
        reason = self._reason("Unsupported weight_bits: 16")
        assert "not a flag" in reason

    def test_it_points_at_the_model_card(self):
        assert "model card" in self._reason("Unsupported weight_bits: 16")

    def test_the_evidence_survives(self):
        assert "Unsupported weight_bits: 16" in self._reason(
            "Unsupported weight_bits: 16")

    def test_a_healthy_launch_gets_no_hint(self):
        from ainode.engine.load_phase import LoadPhaseTracker

        tracker = LoadPhaseTracker()
        tracker.reset()
        for line in ("Loading model from scratch...",
                     "GPU KV cache size: 1,072,101 tokens"):
            tracker.observe(line)
        assert "not a flag" not in tracker.failure_reason()
