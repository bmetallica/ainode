"""The FP4-Local Flash-Next entry: eugr's cluster recipe, on the B12X image."""

from __future__ import annotations

from ainode.core.config import NodeConfig
from ainode.engine.backends.eugr import EugrBackend, _is_safe_env_pair
from ainode.models.registry import CURATED_CLUSTER_MODELS

E = CURATED_CLUSTER_MODELS["qwen3.8-flash-next-fp4-local"]


def _flag(name):
    args = E.extra_vllm_args
    return args[args.index(name) + 1]


def test_it_runs_on_the_b12x_image_across_two_nodes():
    assert E.engine_image_eugr == "vllm-node-b12x"
    assert E.proven_tp == 2 and E.verified is True
    assert E.recommended_gmu == 0.8


def test_the_checkpoints_own_choices():
    assert _flag("--kv-cache-dtype") == "fp8"
    assert _flag("--load-format") == "b12x"
    assert _flag("--speculative_config.num_speculative_tokens") == "4"
    assert "--no-enable-flashinfer-autotune" in E.extra_vllm_args
    assert "ple_table_memory" in _flag("--additional-config")


def test_every_rank_reads_its_own_part_of_the_checkpoint():
    # read_mode "auto" turned into the one-host GDS read on spark-1432 +
    # spark-659b: "shared checkpoint reads require distinct CUDA devices on
    # one host".
    import json

    assert json.loads(_flag("--model-loader-extra-config")) == {"read_mode": "bounce"}


def test_that_failure_is_named():
    from ainode.engine.load_phase import LoadPhaseTracker

    tracker = LoadPhaseTracker()
    tracker.reset()
    tracker.observe("(Worker_TP1) ValueError: shared checkpoint reads require "
                    "distinct CUDA devices on one host")
    tracker.fail("the launcher exited (code 1)")
    reason = tracker.failure_reason()
    assert "read_mode" in reason and "bounce" in reason


def test_no_expert_parallelism_across_nodes(monkeypatch):
    # "NotImplementedError: NVFP4-CSF supports TP without EP/DP"
    import ainode.models.architecture as arch

    class _Facts:
        is_moe = True
        weight_bytes = 1

    monkeypatch.setattr(arch, "facts_for", lambda app, model: _Facts())
    assert arch.architecture_args({}, E.hf_repo, 2) == []
    # Other MoE models keep it.
    assert arch.architecture_args({}, "org/some-moe", 2) == ["--enable-expert-parallel"]
    assert "--enable-expert-parallel" not in E.extra_vllm_args


def test_parallelism_is_left_to_ainode():
    for flag in ("--tensor-parallel-size", "--pipeline-parallel-size", "--port", "--host"):
        assert flag not in E.extra_vllm_args


def test_every_environment_entry_can_reach_the_engine():
    # The eugr launcher expands these unquoted; an unsafe pair is refused.
    for key, value in E.extra_env.items():
        assert _is_safe_env_pair(key, value), key
    args = EugrBackend(NodeConfig(node_id="n", model=E.hf_repo,
                                  extra_env=dict(E.extra_env)))._recipe_env_args()
    assert "B12X_POLICY_MODE=auto" in args


def test_a_typed_drop_still_removes_it(monkeypatch):
    # The recipe merge consumes drop: entries before the architecture flags
    # are added, so they are read from the request itself.
    import ainode.models.architecture as arch
    from ainode.engine.serve_args import body_dropped_flags

    class _Facts:
        is_moe = True
        weight_bytes = 1

    monkeypatch.setattr(arch, "facts_for", lambda app, model: _Facts())
    dropped = body_dropped_flags({"extra_vllm_args": "--x 1 drop:--enable-expert-parallel"})
    assert arch.architecture_args({}, "org/some-moe", 2, dropped=dropped) == []
    assert body_dropped_flags({"extra_vllm_args": ["drop:--load-format"]}) == {"--load-format"}
    assert body_dropped_flags({}) == set()
