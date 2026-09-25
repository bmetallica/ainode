# Coding models for two GB10 nodes

A shortlist, measured rather than remembered. The question it answers: which
modern model is worth loading as *the* coding model across two DGX Spark
(GB10) nodes at `--tensor-parallel-size 2`, serving two parallel sessions with
real context.

Every size below is the Hugging Face Hub's own byte count for the repo
(`safetensors`/`bin` siblings, decimal GB, read 2026-09-24). Every KV figure
comes from the same formula the planner uses,
`ainode/planner/compute.py:kv_bytes_per_token`, at `--kv-cache-dtype fp8`.

That is the conservative column. The engine image here also offers **4-bit**
KV — `nvfp4`, `nvfp4_4over6`, `int4_per_token_head` — which halves the figures
below for a conventionally-attending model. It does **not** help the MLA
entries: `nvfp4_ds_mla` names a layout, not a width, and on DeepSeek-V4 that
layout is the same 584 bytes per token per layer as `fp8_ds_mla` — only the
kernel dispatch differs, and stock vLLM dispatches the 4-bit name to the slow
path, where long-context decode collapses to about a tenth of the throughput.
Serve the MLA models with `fp8_ds_mla`.

(Both facts come from MiaAI-Lab's two-Spark DeepSeek-V4-Flash recipe,
`docs/PATCHES.md` issue #22, MIT — read, not copied.)
Nothing here has been served on this cluster yet — the arithmetic says it
fits, the hardware has not yet said so. That distinction is kept explicit in
the verdict column.

## The budget being spent

Per node, 128 GB unified memory, of which the stack hands out roughly:

| deduction | GB | where it comes from |
| --- | --- | --- |
| OS, page cache floor, container | ~12 | observed `MemAvailable` on an idle node |
| system reserve | 4 | `SYSTEM_RESERVE_GB`, `ainode/planner/compute.py` |
| memory-guard reserve | 4 | `warn_gb - SYSTEM_RESERVE_GB`, `budgets_with_guard_reserve` |
| plan headroom | ~7.7 | `plan_headroom_gb()` — 6 % of total, bounded 1–8 GB |
| **left for weights + KV + activations** | **~100** | |

So two nodes at TP=2 give roughly **200 GB pooled**, minus ~8 GB of
activations, workspace and (with `--enforce-eager`) no graph pool. Call it
**192 GB for weights and KV cache together**.

Two facts shape the whole list:

- **Three nodes, two ranks.** Tensor parallelism needs the attention head
  count to divide the rank count, and three divides almost nothing that
  ships — 16, 20, 32, 40 and 64 are what the checkpoints here actually carry.
  Published three-node recipes for this hardware exist and reach TP=3 by
  patching the engine to pad the heads; that is not a flag, and it is not in
  this image. So a three-node cluster runs one model across two nodes and
  serves something else on the third.

- **MoE is the design point.** GB10 decode is bandwidth-bound at 273 GB/s per
  node, so what matters is *active* parameters per token, not total. A 300B
  MoE with 3B active decodes like a 3B model and remembers like a 300B one.
  All ten entries are MoE except where noted.
- **Experts must be sharded, not replicated.** Tensor parallelism splits
  attention and dense layers but duplicates experts unless
  `--enable-expert-parallel` is passed. `ainode/models/architecture.py`
  emits it automatically for any model whose config declares experts; without
  it the per-node figures below are wrong by roughly a factor of two.

KV turns out not to be the binding constraint for most of these. The hybrid
(linear + full attention) and MLA architectures cache 10–27 kB per token, so
even 2 × 256k tokens costs under 15 GB. The exceptions are called out.

## The ten

Sizes: **disk** is the repo total; **/node** is that divided by two, which is
what TP=2 with expert parallelism actually places on each node.

### 1. `abacusai/Smaug-Flash` — 166.9 GB, 83.5 GB/node

An agentic-coding finetune of `deepseek-ai/DeepSeek-V4-Flash-0731`, same
layout, same mixed block-FP8 attention / packed-FP4 experts, same 1M context,
DSpark multi-token speculative module intact. 43 layers, MLA, 256 routed
experts with 6 + 1 shared active.

KV at fp8: 43 × (512 + 64) = **24.2 kB/token** → ~1.0M tokens pooled →
**2 × 256k sessions with room to spare**.

This is the top pick for one reason beyond the benchmarks: a DeepSeek-V4-Flash
is already what runs on nodes 1 and 2 here, and this is byte-compatible with
it. Same planner numbers, same launch recipe, same context length. The card
claims +14.3 LiveBench agentic-coding and +10.1 Terminal Bench 2.1 over the
base. MIT licensed.

*Verdict:* fits with margin; arch already proven on this cluster.

### 2. `Qwen/Qwen3-Coder-Next-FP8` — 80.4 GB, 40.2 GB/node

80B total, ~3B active, `Qwen3NextForCausalLM` — hybrid attention, one full
attention layer in four (`full_attention_interval: 4`, so 12 of 48 layers
cache). 262k context, 2 KV heads, head dim 256.

KV at fp8: 2 × 12 × 2 × 256 = **12.0 kB/token** → the 112 GB left over after
weights would hold millions of tokens. Context is free here; the cap is the
model's own 262,144.

The throughput pick. 3B active over a 273 GB/s node means it decodes fast
enough to feel interactive with two sessions batched, and it is the most
downloaded coding model on the Hub by a wide margin.

*Verdict:* fits easily; the hybrid linear-attention path is the one thing to
watch — it carries a recurrent state per sequence and has historically been
fussy with `--enforce-eager`.

### 3. `ucbye/Qwen3-Coder-Next-NVFP4-GB10` — 45.8 GB

Same model as #2 in NVFP4, quantised specifically for GB10.
`saricles/Qwen3-Coder-Next-NVFP4-GB10` is the same 45.8 GB with more community
uptake; `RedHatAI/Qwen3-Coder-Next-NVFP4` (47.6 GB) is the vendor-built
alternative.

**This fits on one node.** That is the honest headline, and it argues against
loading it at TP=2 at all: tensor parallelism adds two all-reduces per layer
over the fabric and does not improve single-stream latency for a model that
already fits. Load it TP=1 on one node, keep the other node for the second
session, and you get more total throughput than splitting one copy.

*Verdict:* the best value on the list — but as a one-node model, not a
two-node one.

### 4. `Kwaipilot/KAT-Coder-V2.5-Dev` — 69.3 GB, 34.6 GB/node

35B total / 3B active MoE, built on Qwen3.6-35B-A3B, post-trained with SFT+RL
specifically for agentic coding; the card claims SOTA at its parameter scale
and a near-elimination of malformed tool tags (9.34 % → 0.28 %), which is the
failure mode that actually breaks agent loops. 262k context, text-only — the
vision tower is not in the open-weight release.

KV at fp8: 10 full-attention layers of 40 × 2 heads × 256 = **10.2 kB/token**,
the cheapest on the list.

Quantised: `sakamakismile/KAT-Coder-V2.5-Dev-NVFP4` at 21.9 GB (one node,
easily) or `cyankiwi/KAT-Coder-V2.5-Dev-AWQ-INT4` at 24.4 GB.

*Verdict:* fits anywhere; the best tool-calling behaviour per gigabyte.

### 5. `deepseek-ai/DeepSeek-V4-Flash-0731` — 166.9 GB, 83.5 GB/node

The base that #1 finetunes. Stronger general capability, weaker agentic
coding. Worth keeping as the fallback if Smaug's finetune shows regressions on
non-coding work. Identical footprint and identical KV arithmetic.
`deepseek-ai/DeepSeek-V4-Flash-DSpark` (same 166.9 GB) is the DSpark-packaged
variant already named in the catalog.

*Verdict:* known quantity on this hardware.

### 6. `XiaomiMiMo/MiMo-V2.6-Flash-RL` — 177.7 GB, 88.9 GB/node

FP8, 48 layers, 256 experts / 8 active, 1M declared context. The largest model
on this list that still fits two nodes.

And the one where KV is the binding constraint: 4 KV heads × head dim 192,
conventional GQA, no MLA → 2 × 48 × 4 × 192 = **72.0 kB/token**. After 177.7 GB
of weights only ~14 GB of pooled memory is left, which is ~194k tokens total —
**2 × 96k sessions, not 2 × 256k**, and nowhere near the advertised 1M.

*Verdict:* fits, but pay for the weights in context. Only worth it if the
model is clearly better at your work than #1, which is not established.

### 7. `whitecircle/GLM-4.7-Flash-Coder` — 59.9 GB, 29.9 GB/node

A coder finetune of GLM-4.7-Flash: 31B total / ~3B active, 47 layers, MLA
(`kv_lora_rank` 512 + `qk_rope_head_dim` 64), 202,752 context.

KV at fp8: 47 × 576 = **27.1 kB/token** → ~4.9M tokens pooled. Context capped
by the model, not by memory.

The base model is one of the most-downloaded open models on the Hub;
`GadflyII/GLM-4.7-Flash-NVFP4` (20.4 GB) and `QuantTrio/GLM-4.7-Flash-AWQ`
(19.8 GB) quantise the base, not the coder finetune.

*Verdict:* fits easily on one node in NVFP4; the coder finetune exists only in
bf16, which is why it is listed at 59.9 GB.

### 8. `Jab1718/qwen3.8-flash-coder-85gb-bf16` — 85.2 GB, 42.6 GB/node

`Qwen4ExpForCausalLM`, 48 layers, 160 experts / 10 active, 262k context,
KV **12.0 kB/token**. Explicitly built and sized for an 85 GB budget, which is
a GB10-shaped number.

*Verdict:* fits comfortably. Thin community track record (a few thousand
downloads) — treat as experimental.

### 9. `empero-ai/Qwen3.8-35B-A3B-Distill` — 71.9 GB, 36.0 GB/node

35B / 3B active, same `Qwen3_5Moe` hybrid family as #4, 262k context, KV
**10.2 kB/token**. A general model with strong code rather than a coding
specialist — the one to pick if the same instance also has to answer
non-coding questions. `ornith-ai/Ornith-1.5-35B-A3B` is a same-size, same-arch
alternative.

*Verdict:* fits with enormous margin.

### 10. `IFM/K2-Horizon-MoVA-36B-A4B` — 74.9 GB, 37.5 GB/node

37B / 4B active, 48 layers, 100 experts / 8 active, and the longest declared
context here: **524,288**.

The context is not free. 8 KV heads × head dim 128, plain GQA →
**96.0 kB/token**. Two sessions at the full 512k would want ~100 GB of KV; the
117 GB left after weights does cover it, but it is the only entry where the
planner will have to think about KV at all.

*Verdict:* the long-context specialist. Take it if a single session really
needs half a million tokens; otherwise #1 gives the same reach for a quarter
of the KV.

## Honourable mentions

- `Logics-MLLM/Logics-SWE-Qwen3.6-27B` (54.7 GB) — SWE-specialised, but
  **dense**, so decode is bandwidth-bound at ~27 GB read per token and it will
  feel slow next to any A3B MoE. 128 kB/token of KV, too.
- `cyankiwi/Qwen3-Coder-Next-AWQ-4bit` (48.2 GB) — AWQ alternative to #3 if
  NVFP4 gives trouble.
- `Qwen/Qwen3-Coder-Next` in bf16 (159.4 GB) — the full-precision original.
  Fits two nodes, but there is no evidence the extra 79 GB buys anything over
  the FP8 build.

## What was ruled out, and why

| model | size | reason |
| --- | --- | --- |
| `zai-org/GLM-5.3` | 755.6 GB | needs 8 nodes |
| `XiaomiMiMo/MiMo-V2.6-Pro-RL` | 573.5 GB | needs 6 nodes |
| `deepseek-ai/DeepSeek-V4.1-Flash` | 510.3 GB | needs 6 nodes |
| `FINAL-Bench/Darwin-397B-ZTC` | 418.7 GB | needs 4 nodes; this cluster has 3 |
| `XingChen-AGI/Xing4.0-29B-A4B-FP8` | 33.2 GB | `Xing4_0ForCausalLM` is not in vLLM's model registry |
| `yandex/AliceAI-Foundation-80B-A3B-Base` | 162.6 GB | `AliceAIForCausalLM` is not in vLLM's model registry; base model, not instruct |
| `prism-ml/Ternary-Bonsai-2-27B` | — | gated repo, could not be sized |

Architecture support was checked against
`vllm/model_executor/models/registry.py` on vLLM main. The engine image is
built from eugr's rolling `prebuilt-vllm-current` release
(`scripts/build-base-image.sh`), so it tracks main closely — but a model whose
architecture landed in vLLM after the pinned build will still fail to load.

## Launching any of these

The flags the engine emits automatically, and which matter here:

```
--tensor-parallel-size 2
--enable-expert-parallel        # ainode/models/architecture.py, for any MoE
--kv-cache-dtype fp8            # never on a vision model
--enforce-eager                 # Blackwell/ARM stability
--gpu-memory-utilization <planner's figure>
```

A hardware check after loading one of these, on the head:

```bash
curl -s localhost:3000/api/planner/plan \
  -H 'content-type: application/json' \
  -d '{"hf_repo":"abacusai/Smaug-Flash","node_ids":["<node1>","<node2>"]}' \
  | python3 -m json.tool
```

Success criterion: `"fits": true`, `"tensor_parallel_size": 2`,
`weights_per_node_gb` within a gigabyte or two of the `/node` column above,
and `kv_tokens` at least twice the context you intend to serve.

## Note on getting them here

At 167 GB, #1 and #5 are a long download on a link that has already dropped
mid-transfer once. The import tool exists for exactly this: drop the files on
the head under `/model-import` and take them from the UI, or upload them
through the browser. See the import section in the README.
