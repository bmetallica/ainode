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
    assert E.proven_tp == 2 and E.verified is False


def test_the_checkpoints_own_choices():
    assert _flag("--kv-cache-dtype") == "fp8"
    assert _flag("--load-format") == "b12x"
    assert _flag("--speculative_config.num_speculative_tokens") == "4"
    assert "--no-enable-flashinfer-autotune" in E.extra_vllm_args
    assert "ple_table_memory" in _flag("--additional-config")


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
