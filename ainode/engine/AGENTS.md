# engine/ — AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `Titanium Lab`. Working-state runbook: `ops/runbooks/2026-06-17-235b-moe-tp4-working-state.md`.

## Edit contract — DISTRIBUTED LAUNCH (dangerous, read before any change)

- **Launch distributed serves via the systemd path, NOT the dashboard LAUNCH button / `POST /api/sharding/launch`** (`sharding_routes.py`). That path auto-discovers peers from mgmt-LAN UDP source IPs (`192.168.0.x`), lands a Ray worker on a non-GPU address, and dies with `RuntimeError: current platform does not support ray` (and pollutes Ray with mgmt-IP nodes). Use `config.json` `distributed_mode="head"` + explicit fabric `peer_ips` + `systemctl restart ainode`.
- **`peer_ips` are the fabric (`10.100.0.x`)** — never mgmt LAN or Tailscale addresses. Exception: on a **switchless mesh** the fabric address IS the shared Ethernet (see below), because no CX7 subnet reaches every node.
- **vLLM flags are emitted by the backend, not hand-edited per run.** Change them in `backends/`, not by asking a user to edit a command.
- **Never read `config.cluster_interface` directly in a backend.** Go through `ainode.cluster.topology.topology_for_config(config)` and use `coord_interface` (Ray / SSH / socket env) and `rdma_hcas` (`NCCL_IB_HCA`). On a direct-attach or switched cluster those resolve back to `cluster_interface` and today's local HCA detection, so the 2-/4-node TP setups are unaffected; on a 4-active-link mesh they diverge, and reading the raw field silently pins coordination to a link that reaches one neighbour.
- **Never write an empty `ETH_IF` or `IB_IF` into the launcher `.env`.** `launch-cluster.sh` then falls back to its own autodiscovery, which needs `ibdev2netdev` — not installed in the AINode image. `_write_eugr_env` raises instead; keep that guard.

## vLLM flag invariants (GB10 / Blackwell ARM)

- **Keep `--enforce-eager` — this is the GB10/sm120 fix, not a perf knob.** FlashInfer (vLLM's auto-pick on Blackwell) crashes its prefill kernel (`BatchPrefillWithPagedKVCache`, `illegal instruction`) **under CUDA-graph capture** on GB10 (sm120), killing EngineCore on the **first real prefill**. The engine loads + reports READY, then suicides (vLLM SIGTERMs its own Ray workers) — `/v1/models` 200 is NOT proof of a working engine; verify with a **long-prompt generation**, not the readiness endpoint. `--enforce-eager` disables graph capture and the same kernel runs clean. Re-enabling graphs (for throughput) needs a working non-FlashInfer backend first.
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is set but currently a NO-OP** — this scitrera/vLLM 0.17.1 build ignores it (ranks still log `Using FLASHINFER`). Kept as an env-overridable hedge (correct value may be `TRITON_ATTN_VLLM_V1`); do not rely on it — `--enforce-eager` is what's actually preventing the crash.
- **Never add `--enable-expert-parallel`** — it hangs on this MoE/hardware.
- **`--gpu-memory-utilization` target `0.85`** (config default is `0.9`).
- `--kv-cache-dtype fp8` is required for long context (32k+) or it OOMs.

## Fabric topology (mesh vs. direct)

- `ainode/cluster/topology.py` classifies by **active CX7 link count**: 2 = `DIRECT`, 4 = `MESH`, anything else = `UNKNOWN`. `UNKNOWN` behaves exactly like `DIRECT` — degrade to current behaviour, never refuse to launch.
- **Only `MESH` may change any emitted value.** A change that also fires on `DIRECT`/`UNKNOWN` is a regression against the existing clusters; `tests/test_mesh_fabric.py` guards both sides and any new fabric behaviour belongs there.
- Mesh adds exactly three NCCL vars (`NCCL_NET_PLUGIN=none`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_MERGE_NICS=0`) and hands NCCL **all** RoCE devices. Do not subnet-filter HCAs on a mesh — each device is on its own subnet by design, so any filter cuts the ring down to one cable.

## Don't kill a slow launch

A multi-minute MoE profiling forward-pass with GPUs at 0% and quiet logs is **not** a hang — do not SIGTERM it. (A premature "hung" call cost a whole session; see the runbook.) Wait 3–5 min for `:8000` to bind.

## Verification

- After engine/flag changes: `pytest tests/` and confirm the resolved vLLM command in the launch logs matches the invariants above. Don't claim a serve works unless you saw it reach READY and generate tokens.
