# engine/ — AGENTS.md (edit contract)

Parent: `../../AGENTS.md` · State / "why" / history: Obsidian Vault → `Titanium Lab`. Working-state runbook: `ops/runbooks/2026-06-17-235b-moe-tp4-working-state.md`.

## Edit contract — DISTRIBUTED LAUNCH (dangerous, read before any change)

- **`POST /api/sharding/launch` (the dashboard LAUNCH button) is a supported path.** It once auto-discovered peers from mgmt-LAN UDP source IPs (`192.168.0.x`), landing a Ray worker on a non-GPU address and dying with `RuntimeError: current platform does not support ray`. That is fixed: it resolves each selected member to its **announced** `fabric_ip` and returns 422 rather than falling back to a mgmt address when one is missing. Do not reintroduce a `peer_ip` fallback there. The systemd path (`config.json` `distributed_mode="head"` + explicit `peer_ips` + `systemctl restart ainode`) still works and is the right choice for a fixed, always-on placement; the API path is what the UI uses and the only way to pick a non-tensor split.
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

- **Never spawn `vllm` from AINode's own process.** The orchestrator container is `python:3.12-slim` — no CUDA, no vLLM, no NCCL — so `Popen(["vllm", ...])` fails with `[Errno 2] No such file or directory: 'vllm'`. Both backends run the engine in its own container: eugr through `launch-cluster.sh` (`--solo` for one node, plain for many), nvidia through `docker run`. `_build_solo_cmd` survives only as the argv builder for that script.

## Engine images across nodes

- **Every node runs the engine in a container, so every node needs that image.** `launch-cluster.sh` checks and aborts when one is missing or when the ids differ — the head places it first (`engine/distribute.py`) so that check passes instead of failing.
- **Pull first, copy second.** A registry image is far cheaper pulled on each peer in parallel — layers dedupe — than streamed ~20 GB through the head. The copy is the fallback for a locally built image, and the correction when a pull lands a *different* build of the same tag.
- **Ids must match, not just tags.** A tag that moved in the registry between two pulls leaves ranks on different builds, which fails later and far less clearly.
- This is **not** best-effort, unlike weights: the launch fails anyway without it, so failing here with "could not place `<image>` on `<node>`" is strictly more useful.

## Values that reach launch-cluster.sh (security)

- **Never write an unvalidated string into the launcher `.env` or into `VLLM_SPARK_EXTRA_DOCKER_ARGS`.** Upstream re-quotes every `CONTAINER_*` value by interpolating it into a Python one-liner (`launch-cluster.sh`: `python3 -c "…shlex.quote('$value')…"`), so a single quote in the value closes that literal and the rest runs as code. `VLLM_SPARK_EXTRA_DOCKER_ARGS` is worse: it is expanded **unquoted** into `docker run`, so whitespace injects flags — `-v /:/host` or `--privileged` is a host compromise.
- Device names go through `topology.is_safe_device_name()`, paths through `eugr._is_safe_path_arg()`. Both are enforced at the **sink** (`_write_eugr_env`, `start_distributed`) as well as at `PATCH /api/config`, because `config.json` is also editable by hand.
- This matters because **the API is unauthenticated unless the operator turns auth on**, and `cluster_interface` / `coord_interface` / `rdma_hcas` / `models_dir` are all PATCHable. Adding another PATCHable field that reaches the launcher means adding a validator for it.
- The quoting flaw is upstream's (eugr/spark-vllm-docker, MIT) and is not ours to patch from here. Not feeding it hostile input is.

## Parallelism (tensor / pipeline / data)

- **Never derive the split from the node count in a backend.** `ainode/engine/parallelism.py` owns it: the launch path resolves a `ParallelPlan` and snapshots it into the config; backends read it. A config with no resolved sizes falls back to TP across every node, which is what the backends computed before — keep that fallback, it is what makes an old `config.json` and the systemd launch path keep working.
- **TP is only ever 1, 2, 4 or 8.** Tensor parallelism splits attention heads and head counts are powers of two, so TP=3 has no models behind it. `plan_for()` refuses it with the alternatives named; do not add a bypass.
- **A 3-node cluster means pipeline (or data) parallel.** `auto` resolves to PP there. Upstream's working 3-node recipe is `-tp 1 -pp 3` (`recipes/3x-spark-cluster/`, MIT).
- **Data parallelism is unverified on this hardware** — documented upstream, no recipe behind it. `recommend_strategy()` never picks it; it stays an explicit operator choice. If you verify it, say so here.
- Emit a `--*-parallel-size` flag only when that axis is > 1, so a solo serve and a plain TP launch produce the command line they always did.

## Losing a member node

- A distributed instance whose member node disappears keeps running on the head but has lost the ranks Ray placed there — it **cannot serve**. `/api/cluster/resources` reports it `degraded` with `missing_peer_ips`; the UI badges it amber and offers RELAUNCH.
- **The head never relaunches by itself.** A model spread across three nodes usually does not fit on two, so an automatic retry trades a visible outage for an OOM. `/api/sharding/relaunch` is the explicit action, and it refuses with a reason (bad axis, or the weights no longer fit) rather than trying.
- Relaunch **re-plans** for the smaller node set — a TP=4 instance losing a node comes back as PP=3, never the impossible TP=3 — and delegates placement to `handle_sharding_launch` so there is one launch implementation, not two.
- `save_instance_manifest` still skips distributed instances, so a head restart does not bring them back. That is unchanged and deliberate.

## Fabric topology (mesh vs. direct)

- `ainode/cluster/topology.py` classifies by **active CX7 link count**: 2 = `DIRECT`, 4 = `MESH`, anything else = `UNKNOWN`. `UNKNOWN` behaves exactly like `DIRECT` — degrade to current behaviour, never refuse to launch.
- **Only `MESH` may change any emitted value.** A change that also fires on `DIRECT`/`UNKNOWN` is a regression against the existing clusters; `tests/test_mesh_fabric.py` guards both sides and any new fabric behaviour belongs there.
- Mesh adds exactly three NCCL vars (`NCCL_NET_PLUGIN=none`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_MERGE_NICS=0`) and hands NCCL **all** RoCE devices. Do not subnet-filter HCAs on a mesh — each device is on its own subnet by design, so any filter cuts the ring down to one cable.

## Don't kill a slow launch

A multi-minute MoE profiling forward-pass with GPUs at 0% and quiet logs is **not** a hang — do not SIGTERM it. (A premature "hung" call cost a whole session; see the runbook.) Wait 3–5 min for `:8000` to bind.

## Verification

- After engine/flag changes: `pytest tests/` and confirm the resolved vLLM command in the launch logs matches the invariants above. Don't claim a serve works unless you saw it reach READY and generate tokens.
