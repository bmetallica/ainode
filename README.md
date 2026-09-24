<!--
AINode — local AI platform for NVIDIA GB10 and any NVIDIA GPU server.
Keywords: NVIDIA DGX Spark, ASUS GX10, vLLM, Ray, tensor parallel,
OpenAI-compatible API, local LLM, self-hosted AI, LoRA fine-tuning,
cluster inference, GB10, CUDA 13, NCCL, RoCE, RDMA, container AI
platform, open source ChatGPT alternative.
-->

<p align="center">
  <img src="docs/images/ainode-logo.png" alt="AINode" width="220"/>
</p>

<h1 align="center">AINode</h1>

<p align="center">
  <strong>Turn any NVIDIA GPU into a local AI platform.</strong><br/>
  <em>Inference + fine-tuning in your browser. One container to install. Add nodes, they find each other.</em>
</p>

<p align="center">
  <a href="https://github.com/bmetallica/ainode/releases/latest"><img alt="release" src="https://img.shields.io/github/v/release/bmetallica/ainode?display_name=tag&style=flat-square&color=76B900&label=release"></a>
  <a href="https://github.com/bmetallica/ainode/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-76B900?style=flat-square"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="https://github.com/users/bmetallica/packages/container/package/ainode"><img alt="ghcr" src="https://img.shields.io/badge/ghcr-bmetallica%2Fainode-24292e?style=flat-square&logo=github"></a>
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-13-76B900?style=flat-square&logo=nvidia&logoColor=white">
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-0.19-7C3AED?style=flat-square">
  <img alt="Ray" src="https://img.shields.io/badge/Ray-2.54-028CF3?style=flat-square">
  <a href="https://github.com/bmetallica/ainode/stargazers"><img alt="stars" src="https://img.shields.io/github/stars/bmetallica/ainode?style=flat-square&color=FFD700"></a>
  <a href="https://releasebot.io/updates/bmetallica/ainode"><img alt="Release Bot" src="https://releasebot.io/Full.svg" height="20"></a>
</p>

<p align="center">
  <a href="https://github.com/bmetallica/ainode">github</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/bmetallica/ainode#readme">docs</a>
  &nbsp;·&nbsp;
  <a href="#getting-started--step-by-step">Getting Started</a>
  &nbsp;·&nbsp;
  <a href="#screenshots">Screenshots</a>
  &nbsp;·&nbsp;
  <a href="#state-of-distributed-inference-june-2026">What Works / What Doesn't</a>
</p>

---

## This is a fork

**Upstream: [getainode/ainode](https://github.com/getainode/ainode)** — everything
below is theirs unless noted. Apache-2.0, same as this fork. The UI footer says
so too: upstream's credit stays, with *forked by bmetallica* next to it.

It exists to run AINode on **three DGX Sparks wired in a switchless ring**, which
upstream does not support. Two assumptions in the original break on that topology:

1. **One cluster NIC per node.** `cluster_interface` is a single field that
   simultaneously names the NCCL socket interface, the address the node
   announces, the SSH/Ray target, and the subnet filter for HCA selection. That
   holds while every node shares one ConnectX-7 subnet. In a ring it does not —
   each port is a private link to a *different* neighbour, so no CX7 subnet
   reaches all three nodes.
2. **Tensor parallelism only.** The launch path always built TP = node count.
   TP splits attention heads across ranks and head counts are powers of two, so
   **TP=3 has no models behind it** — three nodes were simply unusable.

What this fork adds, by area. The full inventory, with reasons and file
pointers, is in [`docs/fork-changes.md`](docs/fork-changes.md).

**The fabric** — topology detection (2 active CX7 links = direct-attach, 4 =
mesh; on a mesh, coordination moves to the shared Ethernet while NCCL gets all
four RoCE devices), a second address list so weights travel over a direct RoCE
cable, and **pipeline and data parallel** in the launch path, the state model
and the UI. An impossible split is refused at the API with the alternatives
named rather than failing deep inside vLLM startup.

**Deciding what to run where** — a **launch planner** that reads the
checkpoint's own `config.json` and the nodes' free memory and works out whether
it fits, on which axis, at what context length and for how many concurrent
users, showing its arithmetic; **profiles** ("what this node should be
serving", captured from what is running and applied to converge); and
**placement** ("this model runs on node X, permanently").

**Importing a model by hand** — for a link that cannot carry 86 GB in one
piece. **Models → Import from files** asks what the repo needs and says which
files are missing, with a direct Hugging Face URL for each. Small files can
go through the browser; a whole checkpoint goes in **`/model-import`** on the
head — the installer mounts that directory into the container at the same
path when it exists, so `org--name/` dropped there appears in the panel and
is *moved* (not copied) into place with one click. Either way the finished
model is checked and mirrored to the other nodes, and an incomplete one is
never mirrored: that would spread the problem rather than the model. The file list comes from the
Hub when it can be reached and from the checkpoint's own index when it
cannot — which is what a node with no route has, and enough to finish a
partial download.

**Downloads you can stop and pick up again** — Pause keeps the partial files
and Resume carries on from them (huggingface_hub reuses its own
`.incomplete`), while Cancel still deletes them. And a download that was
interrupted — by a restart, an update, a lost link — is listed as
**Incomplete** rather than as On disk: the check reads the checkpoint's own
shard index *and* its tokenizer files, so a model missing a shard, or holding
`merges.txt` with no `vocab.json`, says so instead of failing minutes into a
launch with a message about safetensors or a backend tokenizer.

**A recipe that reaches the engine** — a curated model's `extra_env` is now
applied by the eugr backend as well as the NVIDIA and diffusers ones. It was
not, so Qwen3.8-27B-NVFP4 got the `--load-format instanttensor` its recipe
chooses and not the `INSTANTTENSOR_BUFFER_SIZE` cap beside it, and failed
every launch on a staging buffer it had been configured not to ask for.

**Reading the checkpoint before the engine does** — tensor parallelism
splits attention and the dense layers and *replicates every expert on every
rank*, so a 129 GB MoE needs 129 GB per node however short the context is:
AINode reads the expert count out of `config.json` and passes
`--enable-expert-parallel` for any multi-node MoE, curated or
self-downloaded. And a checkpoint that states `quant_algo` without
`quant_method` — the key vLLM selects its quantization backend on — is
refused with the reason and a one-key repair offered, rather than being loaded
as if it were not quantized at all and filling the node with experts at full
width. (A `--quantization` flag cannot fix that one: vLLM derives the method
from the config and rejects an argument that disagrees with it.) The repair
runs on every node that holds the weights, because every rank reads its own
copy.

**Not taking the node down** — on GB10 the GPU's memory *is* the host's
memory, so an engine that over-allocates does not get a CUDA error; it starves
the kernel and the node has to be power-cycled. Three things stand in the way,
in this order: `gpu_memory_utilization` is **capped to what is actually free**
on the tightest participating node (it is a share of *total* memory, so 0.97 —
a figure plenty of model cards recommend — means 124 GB of 128 here); an
**admission gate** sits in front of both launch paths; and a **host memory
guard** in its own thread watches `/proc/meminfo`, refuses launches below a
reserve, and stops an engine when memory falls below the line *or* falls
towards it faster than the reserve would last. The planner stops **short** of
that line rather than filling up to it — it used to size the KV cache to
consume every gigabyte down to the guard's reserve, which made a launch that
went exactly to plan one page-cache fluctuation from being killed. On a member node, where a
distributed launch leaves no instance record, it kills the engine container
directly. Launches also run off the event loop, so a loading node no longer
disappears from the cluster.

**Learning from a kill** — when the guard has to stop an engine, that is
written against the model, not only into the log. The next launch of the same
model asking for the same thing is refused with it — *the guard stopped this
here on 23-09 at gpu-memory-utilization 0.85* — and asking for less, for more
nodes, or with a flag the killed launch did not carry lifts the refusal by
itself. When the cause was fixed by something the record cannot see, the
refusal offers to drop it, keeping the measurements, which are still true.
The gate sits in the shared launch core, not only in the HTTP handler, so the
startup replay and a profile cannot walk past it — a model that cannot start
is not retried at every boot. **Launch anyway** in the form's Advanced section
skips all of it for one launch.
**Settings → Memory Guard** lists everything the guard has stopped, fleet-wide,
with an **Unlock** button each — and a blocked model is badged as such on its
card rather than reading "On disk" while every launch of it is refused. A launch that ends on a signal is
translated too: `code -9` is SIGKILL, which no process can catch, so the
engine's own log is a healthy startup right up to the last line.

**Knowing what happened** — per-phase load timings, an **error assistant**
that explains a failure using a model already running, per-instance containers
and logs, and a **measurement store** that records what each launch actually
cost and prefers that to any estimate.

**Watching it** — MQTT telemetry for system, GPU, **RoCE fabric counters**
(`/proc/net/dev` reads zero while RDMA saturates the link), models, engine
internals (KV cache, preemptions, queue depth), the memory guard, transfers,
launch events and log forwarding. Every topic is documented in
[`docs/mqtt-schema.md`](docs/mqtt-schema.md).

**Image generation** — a second engine kind alongside vLLM, which cannot load
a diffusers pipeline at all. Same card, same guard, same profiles. Any model
from the Hub, in any quantisation that loads: the two catalog entries are
suggestions, and the checkpoint itself decides which engine serves it. See
[`images.md`](images.md).

**Finding a model at all** — the Hub search knows what this deployment can
serve and what it cannot. Filter by kind, and where something will not run,
the card says why in a sentence rather than going grey.

**Updating** — this cluster builds on the head rather than pulling a published
image, so the update button does what an operator would: `git pull` on the
checkout, then `scripts/update-cluster.sh` to build, distribute and restart.
It checks hourly whether *your own fork's* branch is ahead of the commit this
image was built from — a version number moves far less often than the code —
and says so on the dashboard.

**Self-contained distribution** — its own images, installer and CI, so it does
not depend on upstream's releases.

The network heuristics and the 3-node parallelism constraints are adapted from
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT), whose
`autodiscover.sh`, `launch-cluster.sh` and `docs/NETWORKING.md` document this
hardware. Borrowed logic is marked at each site; there are no verbatim copies.

Step-by-step setup for the reference cluster (`ai-vkv`, three Sparks in a ring),
in German, with that cluster's real addresses:
[`docs/mesh/ANLEITUNG-3-NODE-MESH.md`](docs/mesh/ANLEITUNG-3-NODE-MESH.md).

Background on what was changed and why: [`docs/mesh/PHASE1-ANALYSE.md`](docs/mesh/PHASE1-ANALYSE.md)
(an audit of the original code before any of it was touched — German) and
[`docs/mesh/BOOTSTRAP.md`](docs/mesh/BOOTSTRAP.md) (publishing this fork's image).

**If you are not running a multi-node GB10 cluster, upstream is the better
choice** — it is the maintained project, and the fabric work here is of no use
to you. Quite a lot of the rest is not hardware-specific, though: the planner,
profiles, placement, the load timings, the error assistant and the measurement
store would work on any cluster, and
[`docs/fork-changes.md`](docs/fork-changes.md) says which pieces those are so
they can be carried back rather than kept here. Two plain bug fixes found along
the way belong upstream outright: a head that crash-looped when a peer was
offline, and a completed model download that reported stale progress because
its poller raced the completion write. Both are in the
[changelog](CHANGELOG.md).

---

## What AINode is

AINode is a self-hosted AI appliance for **NVIDIA GB10** (DGX Spark, ASUS
GX10) and any NVIDIA GPU box. It ships as **one container** that bundles:

- A modern web UI (chat, cluster topology, server console, downloads, training)
- An OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- A GB10-patched vLLM with Ray for cross-node tensor/pipeline parallel
- UDP node discovery for automatic clustering
- NFS-shared model storage so you download once and use everywhere
- Scripted fine-tuning (LoRA, QLoRA, full FT, DPO, distributed DDP)

One `docker pull`, one systemd unit per box, done. No host Python venv,
no source-built vLLM, no fragile runtime wiring.

```bash
curl -fsSL https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh | bash
```

---

## Screenshots

### Cluster view — 4 nodes, 487 GB aggregated VRAM

![Cluster view](docs/images/cluster-4node.gif)

The "MASTER" node (head) runs the API and orchestrates. The smaller
orbiting node (member) has its GPU reserved for a Ray worker that the
head placed. The instance card shows **DISTRIBUTED · TP=2** — the model
is sharded across both GPUs.

### Chat

![Chat](docs/images/chat.png)

Full-featured chat with streaming tokens, prompt history, code
highlighting, per-message metrics (TTFT, tokens/sec, total tokens).
Works against whatever model the cluster has loaded — solo or sharded.

### Server — API console (LM Studio style)

![Server view — API console](docs/images/server-api-console.png)

Live developer console: which models are loaded on which node,
OpenAI-/LM-Studio-/Anthropic-compatible endpoints, per-request logs
with status codes and latency, eject-model buttons, copyable cURL
snippets.

### Downloads — live HF catalog

![Model downloads](docs/images/downloads.png)

Browse trending HuggingFace models, with **AVAILABLE** / **FITS GPU**
badges computed from your cluster's aggregate VRAM. Queue downloads to
the shared NFS cache; any node can load them instantly.

### Training — overview

![Training overview](docs/images/training-overview.png)

Three quick-start paths: **LoRA** (lightweight, most users), **Distributed
DDP** (multi-node fine-tuning), **Full fine-tune** (single large-memory
node). Track active + completed runs, GPU-hours, and jump into dataset
management.

### Training — templates

![Training templates](docs/images/training-templates.png)

Starter recipes for instruction tuning (Alpaca), chat fine-tuning
(ShareGPT), classification heads, DPO / preference learning, and
multi-node DDP. Each template ships a working dataset schema so you
can start training in minutes.

### Config — cluster

![Config — cluster](docs/images/config-cluster.png)

Pin the node's role (`auto` / `master` / `worker`), set a shared
`cluster_id` so only matching nodes see each other, and inspect the
current member list with per-node role, address, and last-seen.

---

## Getting Started — step by step

### Single node (solo mode)

1. **Install Docker** + NVIDIA container toolkit on your Linux box.
2. **Pull the image** and wire up the systemd unit:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh | bash
   ```
   That one-liner:
   ```
   # resolves the highest numeric GHCR tag (never a floating :latest)
   docker pull ghcr.io/bmetallica/ainode:<latest-release>
   # pins it to ~/.ainode/image.env and installs a swappable systemd unit
   systemctl enable --now ainode.service
   ```
   The unit reads the pinned image from `~/.ainode/image.env`
   (`EnvironmentFile`, `Restart=always`), so it survives cold power
   cycles and **replays the models you had loaded** on boot.
3. **Open the UI** at `http://<your-ip>:3000`. First-run onboarding walks
   you through picking a model. Click a model card → click **Launch** →
   chat.

Upgrade is `ainode update` (resolves + pulls the newest pinned release
and restarts) — or `ainode update 0.5.2` to pin a specific version.

**Prefer to pull the image yourself?** GHCR is the only registry this fork
publishes to; Docker Hub mirroring is opt-in and off by default (see
[`docs/mesh/BOOTSTRAP.md`](docs/mesh/BOOTSTRAP.md)):

```bash
docker pull ghcr.io/bmetallica/ainode:latest     # always newest
# pin a release instead: ghcr.io/bmetallica/ainode:0.6.0
```

### Two nodes (distributed mode)

For models that don't fit on one GPU — e.g. a 70B-class model sharded
across two DGX Sparks:

1. **Wire a clean high-speed link** between the two nodes (direct QSFP
   cable on its own `/24`, or a dedicated switch port). See
   [Networking requirements](#networking-requirements) — this matters.
2. **Install AINode on both** (step 1 above).
3. **On the peer**, set member mode in `~/.ainode/config.json`:
   ```json
   {
     "distributed_mode": "member",
     "cluster_interface": "enp1s0f0np0",
     "ssh_user": "sem"
   }
   ```
   `sudo systemctl restart ainode`.
4. **On the head**, set head mode and add passwordless SSH to the peer:
   ```json
   {
     "distributed_mode": "head",
     "peer_ips": ["10.0.0.2"],
     "cluster_interface": "enp1s0f0np0",
     "ssh_user": "sem"
   }
   ```
   ```bash
   ssh-copy-id sem@10.0.0.2 && sudo systemctl restart ainode
   ```
5. **Open the head UI** — you should see both nodes, aggregated VRAM
   ("2 nodes · 244 GB · 2 GPUs"), and the instance badged as
   **DISTRIBUTED · TP=2**.

Want to do it from the browser instead? Open the Launch Instance
panel, pick the model, set **Minimum Nodes=2**, click **Tensor** →
**LAUNCH**. The UI writes the config and hot-swaps the engine for you.

---

## Quantize a model (AWQ / NVFP4)

AINode can compress a full-precision model to 4-bit **in the browser**, on your
own GPU — no external service. Open **Training → Quantize a Model**:

1. **Base model** — a Hugging Face repo id (`Qwen/Qwen3.5-4B`) or an installed model.
2. **Scheme** — **AWQ** (W4A16, proven on GB10 via `awq_marlin`) or **NVFP4**
   (Blackwell-native 4-bit float).
3. **Calibration samples** — default 256 (from `HuggingFaceH4/ultrachat_200k`).
4. *(optional)* **Push result to Hugging Face** — requires a **write** token;
   pushes a private repo under your namespace.

The target node must be **idle** — quantization needs the full unified memory, so
AINode refuses to start a quant job while a model is loaded (unload first, or pass
`force=true`). The output lands in **Installed** as `<org--name>-<scheme>`, ready
to serve.

> AWQ is the proven path on GB10. NVFP4 quantization is newer; **NVFP4 on
> multimodal models (e.g. Qwen3.5) is experimental and not yet verified** — prefer
> AWQ for the Qwen3.5 family today.

**Hugging Face tokens (read vs write).** AINode keeps credentials in a local
Secrets store (`~/.ainode/secrets.json`, mode 0600, obfuscated at rest) with two
HF slots: a **read** token (download gated models) and a **write** token (push to
the Hub — read-only tokens are rejected before any multi-GB transfer). Set them in
**Config → Secrets** (each has a **Test** button showing the detected scope), or
set the read token with `ainode config --hf-token hf_xxx`.

---

## Features

| Feature | Status |
|---|---|
| One-command install | ✅ |
| Unified container image (UI + engine) | ✅ v0.4.0 |
| Auto-detect GPU and memory | ✅ |
| Chat UI in your browser | ✅ |
| OpenAI-compatible API | ✅ |
| Embeddings endpoint (`/v1/embeddings`) | ✅ |
| Live HF model catalog with trending + download manager | ✅ |
| NFS-shared model storage across cluster | ✅ |
| Multi-node auto-discovery (UDP broadcast) | ✅ |
| Distributed tensor-parallel inference across nodes | ✅ (4-node verified — 487 GB aggregated VRAM) |
| Cluster topology UI (members, VRAM aggregate, instance badges) | ✅ |
| Browser-based fine-tuning (LoRA / QLoRA / Full + DDP) | ✅ |
| Training artifact retrieval + download via API | ✅ |
| LoRA adapter merge into base model | ✅ |
| Checkpoint resume | ✅ |
| Evaluation loop (configurable train/eval split) | ✅ |
| W&B logging integration | ✅ |
| Custom training template persistence | ✅ |
| Prometheus metrics endpoint (`/metrics`) | ✅ |
| `ainode role master\|worker\|solo` CLI | ✅ |
| Worker nodes start instantly — no model required | ✅ |
| Web portal available immediately on start | ✅ |
| Cluster-wide update from master UI (`⬆ Update all` button) | ✅ |
| Topology loading animation + per-node fade-in | ✅ |
| AWQ models on GB10 (sm_12.1) — `awq_marlin` kernel fix | ✅ |
| In-browser quantization (AWQ W4A16 / NVFP4) → serve or push to HF | ✅ v0.4.44 |
| Push quantized / fine-tuned models to Hugging Face (write-token) | ✅ |
| Secrets store (HF read + HF write + NGC + W&B + OpenAI), masked + testable | ✅ |
| Federated master router — route `/v1/*` by model name across the cluster | ✅ |
| Load / unload any model on any node from the master UI | ✅ |
| Model stacking — N concurrent models per node, persisted + replayed on boot | ✅ |
| Serve models from on-disk weights (`~/.ainode/models/<slug>`) | ✅ |
| fp8 KV-cache default on GB10 (long-context headroom) | ✅ |
| Per-load overrides (`served_model_name` / `max_model_len` / `kv_cache_dtype` / `quantization` / `trust_remote_code`), persisted across restarts | ✅ v0.5.0 |
| Node-targeted model load (`POST /api/cluster/load {node_id}`) | ✅ v0.5.1 |
| Stacked-load admission guard — explicit `gpu_memory_utilization` required, reject > 0.9 projected total (409) | ✅ v0.5.1 |
| VLM (vision) support — fp8 KV auto-skipped on GB10; `kv_cache_dtype=auto` per-load override | ✅ v0.5.1 |
| LoRA / QLoRA training **and** adapter merge run in a spawned GPU container (slim orchestrator has no torch) | ✅ v0.5.0 |
| Deploy pipeline — `git tag` → CI (self-hosted Spark runner) → GHCR → `ainode update` / cluster update-all (genuine pull + swap) | ✅ v0.5.0 |
| Cancellable, commit-pinned, parallel model downloads | ✅ v0.5.2 |
| Delete a downloaded model from disk (`delete-repo`, frees GB) | ✅ |
| AutoData — Δ-filtered synthetic-data generation (v2.2 val-set lift objective) | ✅ v0.5.0 |
| Proven catalog recipes applied to **distributed** launches too (engine image, parsers, spec-decode) | ✅ |
| Profiles — describe several models as one deployment, apply it, restore it at startup | ✅ |
| Tool calling configured automatically — parser derived from the model family, overridable per launch | ✅ |
| MQTT telemetry — node (CPU/RAM/disk/per-NIC load) + cluster + per-model speed, configured in the UI | ✅ |
| One-command cluster update (`scripts/update-cluster.sh`) — build, distribute, restart, verify; repeatable | ✅ |
| One-command diagnostic report (`scripts/diagnose.sh`) — read-only, secrets redacted, peers included | ✅ |
| Image cache on the head — Hub pull-through + local registry, so an image is fetched once, not once per node | ✅ |
| Launch planner — reads the checkpoint's `config.json` and each node's free memory; answers fit, axis, `max-model-len`, KV, concurrency, and shows its arithmetic | ✅ |
| Host memory guard — own thread, `/proc/meminfo`, refuses launches below a reserve and stops an engine when memory drops below the line or falls towards it too fast; kills the engine container on a member node, which has no instance record; per-node settings with a DGX Spark preset | ✅ |
| Utilization cap — `gpu_memory_utilization` lowered to what is free on the tightest participating node, because on unified memory it is a share of *total* memory and the guard cannot outrun a KV allocation | ✅ |
| Planning headroom — the plan stops short of the guard's line instead of filling up to it, so a launch that goes exactly to plan is not one fluctuation from being killed | ✅ |
| Admission gate in front of **both** launch paths, scoped to the node the instance will run on | ✅ |
| Launches run off the event loop — a loading node no longer drops out of the cluster | ✅ |
| Per-phase load timings — where a five-minute launch actually went | ✅ |
| Error assistant — explains a failed launch using a model that is already serving; the raw error is never replaced | ✅ |
| Measurement store — records what each launch really cost and prefers it to any estimate | ✅ |
| Persistent per-model placement ("this model runs on node X") | ✅ |
| RoCE fabric telemetry — throughput and error counters from `/sys/class/infiniband`, which `/proc/net/dev` cannot see | ✅ |
| Engine telemetry — KV-cache fill, preemptions, queue depth, TTFT, per instance | ✅ |
| MQTT availability via last will, log forwarding, launch events, transfer progress | ✅ |
| Image generation — diffusers engine alongside vLLM, OpenAI `/v1/images/generations`, same card and guard | ✅ |
| Per-instance engine containers, launch scripts and log files | ✅ |
| Client config generator (OpenCode) built from what is actually serving | ✅ |
| Source update from the UI — `git pull` then `scripts/update-cluster.sh`, against your own fork's branch, with the peers' SSH ids held in Settings | ✅ |
| Update check against the commit the image was built from, hourly plus a manual button, with a dashboard banner when the branch is ahead | ✅ |
| Hugging Face search by kind — chat / vision / image generation / embeddings; the kind changes the query, not just the result list | ✅ |
| Per-kind format verdict with its reason — "GGUF holds the transformer alone", "no aarch64 build of Nunchaku's kernels" — instead of a dimmed card | ✅ |
| Embedding models findable in the Hub search (they were excluded, correctly for vLLM and wrongly for a node that serves them in-process) | ✅ |
| Any quantisation that can be loaded: fp8, int4, AWQ, GPTQ, NVFP4, bf16 — the line is the layout, not the bit width | ✅ |

---

## Relation to the Community

AINode builds on excellent open-source work in the DGX Spark ecosystem.
In particular, our base image inherits the patched NCCL from
**[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)**
(`dgxspark-3node-ring` branch), which we've found to be the most reliable
variant for handling GB10 unified-memory topologies and fabric setups.

eugr's project remains the go-to for raw, high-performance vLLM clustering
on Spark hardware. AINode layers a modern browser UI, one-command
deployment, in-browser chat + OpenAI API, and distributed fine-tuning on
top of that strong foundation.

Huge thanks to eugr and the contributors making multi-node Spark setups
practical.

---

## State of Distributed Inference (June 2026)

We owe readers the honest picture, not a checkmark-soup. Here's what's
really running on our hardware.

### What works today (verified)

- **Single-node inference** on any NVIDIA GB10 box (DGX Spark, ASUS GX10).
- **Two-node tensor-parallel** (TP=2) with one GPU per node on a
  direct-connect QSFP `/24`. Both GPUs show ~61 GB of
  `ray::RayWorkerWrapper` memory; NCCL chose `NET/IB RoCE @ 200 Gb/s`.
- **Four-node cluster** (3× DGX Spark + 1× ASUS GX10) — 487 GB
  aggregated VRAM, all four discovered automatically via UDP, topology
  visible in the browser UI. Verified April 2026.
- **One-container-per-node install** — `curl -fsSL https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh | bash -s -- --job worker`
  installs in seconds with no model required.
- **`ainode role`** CLI sets master/worker/solo instantly.
- **Worker nodes start immediately** — no model download, no engine
  warmup. Web portal is up within seconds of `systemctl start ainode`.
- **Shared model storage over NFS** from an NVMe-oF-backed master.
- **UDP cluster discovery** on port 5679 with real peer-IP capture.
- **Inference throughput:** ~35 tok/s for a warmed-up model over
  the RoCE fabric.

- **Four-node TP=4** — verified live on frontier MoE: `nvidia/Qwen3-235B-A22B-NVFP4`
  served at TP=4 across 4× GB10 (~16–17 t/s single-stream, survived a 3,513-token
  prefill). The GB10 sm120 fix was `--enforce-eager` (vLLM's FlashInfer prefill
  kernel emits an `illegal instruction` under CUDA-graph capture on GB10).
- **Federated serving** — a master routes `/v1/*` to the node holding each model;
  models load/unload per node from the browser.
- **Model stacking** — multiple models per node, persisted and replayed on boot.
- **In-browser quantization** — AWQ and NVFP4 jobs run on an idle node and land
  the result in Installed (optionally pushed to Hugging Face).

### What still needs care

- **Ray over Tailscale** — use physical cables or a dedicated switch.
- **Single NIC per cluster subnet** — multi-NIC ambiguity still breaks the NCCL ring.

### Lessons learned the hard way

0. **Role clarity eliminates half the problems.** The single biggest
   UX improvement was `ainode role master|worker|solo`. Workers don't
   need a model, don't need to think, don't need config editing. They
   start in 3 seconds and announce themselves. The master is the only
   node that needs a model. Everything else follows from that.

1. **Single NIC per cluster subnet.** Multi-NIC ambiguity breaks NCCL
   ring setup silently; `NCCL_SOCKET_IFNAME` only tells NCCL which
   address to *listen* on, not which source the kernel picks for
   outbound traffic.
2. **Ray placement groups outlive SIGKILL.** Hung vLLM doesn't release
   the reservation; Ray's GCS still thinks the GPU is busy. Always
   `docker rm -f` the full chain before retrying.
3. **Block-level shared storage is unsafe for multi-writer.** NVMe-oF +
   ext4 mounted on two hosts corrupts under concurrent writes. Put NFS
   on top of a single-host mount.
4. **The patched NCCL in `eugr/spark-vllm-docker`** (`dgxspark-3node-ring`
   branch) is the only variant we've seen reliably handle GB10
   unified-memory topologies. Our base image inherits it.
5. **SSH from a root container into a host user** fails silently when
   keys are mounted read-only from the host. Our entrypoint copies
   `/host-ssh` → `/root/.ssh` with correct perms and injects
   `User <ssh_user>` for peer IPs.

### Why 3 nodes is harder than 2 (and why 4 is probably easier)

**Two nodes**: a single direct-connect cable on one `/24`. One cable,
one subnet, one candidate interface per host. NCCL can't get confused.
TP=2 splits the weights evenly. Solved problem.

**Three nodes**: no simple physical topology. Options:

- **Triangle mesh** (A↔B, B↔C, A↔C) with each link on its own `/30` —
  community tooling assumes this, nobody autoconfigures it.
- **Dedicated cluster switch** with one NIC per node on an isolated
  subnet — easier, but a hardware purchase.
- **Star topology** — asymmetric latency, not recommended.

If your three nodes just share a regular LAN, you hit multi-NIC routing
ambiguity (lesson #1). We did. NCCL ring setup succeeded; data never
flowed.

**Four nodes**: paradoxically simpler once you commit to a switch,
which is the only practical option for 4+. One NIC per node on a fresh
`/24`, TP=4 lines up with vLLM's defaults, and the community has
published recipes (eugr's `recipes/4x-spark-cluster/`, NVIDIA's internal
4× Spark reference setups).

**Our hypothesis:** the difficulty is not *N* nodes — it's *how you
wire N nodes*. Two is forced (one cable). Three forces a topology
decision. Four+ forces a switch, which is what the community tools
expect. Stick to 2 now; buy the switch; jump straight to 4.

---

## Networking requirements

AINode relies on NCCL for cross-node tensor-parallel, and NCCL works
best when it owns a clean link.

- **Passwordless SSH** from the head's host user to every peer.
- **Single active NIC per cluster subnet** on every node. Multiple
  interfaces on the same `/24` breaks the NCCL ring.
- **No VPN between nodes for cluster traffic.** Tailscale is fine for
  laptop→cluster SSH; not fine as the NCCL transport.
- **Consistent MTU** across the cluster subnet.

### Recommended topologies

| Cluster size | Topology | Notes |
|---|---|---|
| 2 nodes | Direct QSFP cable, each end on its own IP in a fresh `/24` | Simplest, verified |
| 3 nodes | Triangle direct-connect (3 cables, each on a `/30`) **or** dedicated switch | Mesh is finicky; switch is easier |
| 4+ nodes | Dedicated QSFP switch on its own `/24`, one NIC per node | The only practical option |

### Diagnostic commands

```bash
# Confirm RDMA is live on your ConnectX-7
ibstat mlx5_0 | grep -E "State|Rate"         # expect "Active" + "Rate: 200"

# Confirm exactly one interface has an IP on the cluster subnet
ip -br -4 addr | grep 10.0.0                 # expect one line per node

# Confirm cross-node reach on the cluster subnet (not Tailscale)
ping -c 2 10.0.0.2
traceroute 10.0.0.2                          # 1 hop = right link

# Passwordless SSH works
ssh sem@10.0.0.2 true && echo OK

# After launching distributed: confirm NCCL uses RoCE, not Socket
docker exec vllm_node bash -c 'grep -E "Using network|NET/IB.*RoCE" \
  /tmp/ray/session_latest/logs/worker-*-01000000-*.out | head -5'
# Expect: "Using network IB" + "NET/IB ... mlx5_0:1/RoCE ... speed=200000"
```

### Optional: GPU Direct RDMA (GDR)

Without GDR, traffic goes GPU → CPU → NIC → NIC → CPU → GPU. With GDR
it bypasses the CPU hop. On GB10 the unified-memory CPU hop is cheap,
so the win is smaller than on discrete GPUs but still measurable.

```bash
# Load the peermem module on each host (not the container)
sudo modprobe nvidia_peermem
echo nvidia_peermem | sudo tee -a /etc/modules-load.d/nvidia-peermem.conf

# Verify NCCL picks it up next launch
docker exec vllm_node bash -c 'grep "GPU Direct RDMA" \
  /tmp/ray/session_latest/logs/worker-*-01000000-*.out'
# Expect: "GPU Direct RDMA Enabled"
```

---

## Shared model storage across a cluster

Downloading a 70 GB model three times on a three-node cluster is
wasteful. AINode supports a shared `models_dir` so every node pulls
from the same cache.

Block-level shared storage (NVMe-oF, iSCSI, Fibre Channel) is fast but
**unsafe for multiple Linux kernels writing simultaneously** — ext4
/ XFS have no distributed lock manager. Layer NFS on top:

```
  Storage array (NVMe-oF, SAN, local NVMe)
         │
         ▼
  MASTER NODE  ← ext4/XFS mounted here, owns the disk
    │   │
    │   └── NFS server exports /mnt/ai-models
    ▼
  WORKERS      ← mount the NFS share at /mnt/ai-shared
```

NFS over a 100G fabric gives 3–8 GB/s — vLLM model loading is a
one-shot sequential read, so you won't notice. For 100 GB+ models
where load time hurts, add an rsync-to-local staging step.

---

## CLI reference

The installer puts a thin `ainode` wrapper at `/usr/local/bin/ainode`.
Host-side commands (`update`) run directly; everything else is forwarded
into the running container via `docker exec`. You never need to type
`docker` yourself.

```bash
ainode update [version]      # resolve/pull newest (or pinned) tag + restart (upgrade in place)
ainode start                 # Start AINode (inference + web UI)
ainode stop                  # Stop AINode
ainode status                # Show cluster status
ainode models                # List available models
ainode service install       # Install the systemd unit
ainode service status        # Show systemd state + recent journal
ainode config                # Show current configuration
ainode logs -f               # Tail the engine log
```

### Updating AINode

Releases ship through a **tag-triggered pipeline**: `git tag vX.Y.Z` →
CI on a self-hosted Spark runner builds and pushes
`ghcr.io/bmetallica/ainode:X.Y.Z`. To upgrade a node in place:

```bash
ainode update
```

That resolves the **highest numeric GHCR tag** (never a floating
`:latest`), pulls it, pins it to `~/.ainode/image.env`, and restarts the
systemd service. Your config (`~/.ainode/config.json`), models
(`~/.ainode/models/`), and fine-tune outputs are on the host — the
container is stateless, so upgrades never touch your data.

To pin a specific version:

```bash
ainode update 0.5.2
```

To roll every node in a cluster from the master, use the **Update all**
button or:

```bash
curl -X POST http://<master>:8000/api/cluster/update-all   # genuine pull + swap on every node
```

#### Updating from your own source, from the UI

A cluster that **builds on the head** instead of pulling a published image
updates differently: **Settings → Updates** runs `git pull` on the checkout and
then `scripts/update-cluster.sh`, and the dashboard says when your fork's
branch is ahead of the commit this image was built from (checked hourly, plus a
**Check for updates** button). The peers it rolls are the SSH ids you enter
there — `Spark2, Spark3` for the reference cluster.

For this to work the container has to be able to see the checkout, so the
installer mounts it at `/ainode-src` when `/opt/ainode` (or `AINODE_SOURCE_DIR`)
is a real repository. An existing node gets that mount by **re-running the
installer** — it is idempotent and keeps `config.json`:

```bash
cd /opt/ainode && git pull
curl -fsSL https://raw.githubusercontent.com/bmetallica/ainode/main/scripts/install.sh | bash
```

`ainode service install` is not the way to do it: on the host `ainode` is a
wrapper into the container, and the unit is written by the installer.

A code-only update rebuilds only the source layers — the commit SHA is
introduced after the dependency install, so a one-line change no longer
re-downloads torch and cuDNN. The build also needs BuildKit, which the host
has and the container did not until the image that carries
`docker-buildx-plugin`; `build-ainode-image.sh`
falls back to a BuildKit-free copy of the Dockerfile when the plugin is
missing, so an older image can still build its successor.

The orchestrator image carries `git` only from the release that introduced
this feature, so an older image cannot run the update that would replace it —
the panel says so rather than failing at `git`. Bootstrap it once from the
head's shell:

```bash
cd /opt/ainode && git pull
scripts/update-cluster.sh --nodes Spark2,Spark3
```

The checkout stays yours: the container runs as root, so `git pull` is run as
the directory's owner rather than being told to ignore the ownership — a pull
as root leaves objects under `.git` that your own shell then cannot write
over.

**Watching it from a shell.** The run is not tied to the browser that started
it: reload, close the tab, come back on another machine — the panel picks the
output back up, and the dashboard carries a banner while one is in flight. From
the head's shell, either of:

```bash
# one line: running / done / failed, when, and from which commit
curl -s localhost:3000/api/update/status |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["summary"])'

docker logs -f ainode | grep --line-buffered 'update |'      # every line, live

curl -s localhost:3000/api/update/status |                   # the last lines
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["summary"]); print(*d["lines"][-30:], sep="\n")'
```

`summary` carries the timestamp and the commit on purpose: a bare `failed` is
indistinguishable from a failure half an hour ago that has since been fixed,
and what `/api/update/status` returns between runs is the *last* one, not a
current one. It is finished when it reads `done` — or when this node restarts
under you, which is the last thing a successful update does. The outcome is
kept in `~/.ainode/update-last.json` and shown in the panel afterwards.

From then on the button does the same thing: `git pull`, build, distribute,
restart the peers, verify them, and restart the head last — which from inside
the container is a `docker stop` of itself that systemd turns into a start on
the new image. The page you started it from goes away with it, so the outcome
is written to `~/.ainode/update-last.json` and shown when you come back.

Re-running the installer never needs the registry if this node already has an
image: when the GHCR package is private or unpublished, it keeps the image
recorded in `~/.ainode/image.env` and only re-renders the unit. Upgrading the
image itself stays a separate, explicit step (`ainode update`).

---

## API

AINode exposes an OpenAI-compatible API. Drop it into any tool that
speaks OpenAI:

Point it at **port 3000**, the one AINode itself answers on. Port 8000 is the
engine's own listener: a vLLM serving one model, with no idea the rest of the
cluster exists, no `/v1/images/generations`, and no routing. Everything below
— federation, the kind filter, image generation — is AINode's, and AINode is
on 3000.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="not-needed",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Works with Open WebUI, LiteLLM, LangChain, llama.cpp clients, and
anything else that speaks OpenAI.

#### `/v1/models` lists what you can chat with

Plain `GET /v1/models` returns the chat-capable models across the whole fleet.
Embedding and image models are not in it: they answer at `/v1/embeddings` and
`/v1/images/generations`, and every OpenAI client — Open WebUI included —
offers whatever this endpoint returns as something to chat with. Ask for them
explicitly:

```bash
curl localhost:3000/v1/models                    # chat + vision
curl localhost:3000/v1/models?type=embedding     # for a RAG client
curl localhost:3000/v1/models?type=image
curl localhost:3000/v1/models?type=all           # the complete inventory
```

Every entry carries `ainode_kind` alongside the OpenAI fields, so a client
that reads `?type=all` can sort them itself.

#### Image generation from an OpenAI client

`POST /v1/images/generations` takes the OpenAI body — `prompt`, `size`, `n`,
plus `steps`, `negative_prompt`, `guidance_scale` and `seed` — and answers
with `data[].b64_json`. It is routed to whichever node holds the image model,
the same way a chat request is.

In **Open WebUI**: Settings → Images → *Image Generation (Experimental)*,
engine **OpenAI**, API Base URL `http://<head>:3000/v1`, any non-empty API
key, and the model typed in by name (`Rin247/Qwen-Image-2.1-FP8`, say) —
plain `/v1/models` deliberately does not offer image models as things to chat
with, so the picker will not list it for you.

```bash
curl -X POST localhost:3000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Rin247/Qwen-Image-2.1-FP8","prompt":"a lighthouse at dusk",
       "size":"1024x1024","steps":20}' |
  python3 -c 'import base64,json,sys; open("out.png","wb").write(
      base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

### Profiles — one deployment, saved and restored

A profile is the set of models the **cluster** should be serving, with
placement and per-load settings. Applying one converges onto it: missing models
are started one after another — each on the node its entry names — and models
that should not be running *here* are stopped here, including a local copy of
something the profile places on another node. A default profile is applied at
startup and replaces the instance-manifest replay, which could only record
single-node instances.

Placement is exclusive. An embedding model assigned to node 3 is unloaded from
every other node when the profile is applied, and an embeddings request for it
that reaches another node is answered by node 3 rather than by loading a second
copy where it arrived.

```bash
# save what is running right now
curl -X POST localhost:3000/api/profiles/capture \
  -H 'Content-Type: application/json' \
  -d '{"name":"Production","description":"chat + code + embeddings"}'

curl localhost:3000/api/profiles                      # list, with the default marked
curl -X POST localhost:3000/api/profiles/Production/apply   -d '{}'
curl -X POST localhost:3000/api/profiles/Production/default -d '{}'
```

Everything is also in the **Profiles** tab of the web UI, which is where the
buttons that call these live.

### Telemetry over MQTT

A scrape needs the monitoring host to reach every node; a publish needs each
node to reach one broker. On a switchless mesh that difference matters, and
most labs already run a broker behind Home Assistant, Node-RED or Telegraf.

Configured in **Config → Monitoring** (broker, credentials, topic prefix,
interval), with *Test connection*, *Publish now* and a payload preview. The
password is kept in the secrets store, not in `config.json`.

```
<prefix>/<node-id>/status            online / offline — retained, set by the broker's last will
<prefix>/<node-id>/system            CPU, load, memory, disk, per-interface Mbit/s and % of link, temperatures
<prefix>/<node-id>/gpu               utilization, memory, temperature
<prefix>/<node-id>/fabric            RoCE links: throughput and error counters, which /proc/net/dev cannot see
<prefix>/<node-id>/models            what is loaded, requests, errors, latency percentiles, per-model speed
<prefix>/<node-id>/engine/<model>    from vLLM itself: KV cache, preemptions, queue depth, TTFT
<prefix>/<node-id>/safety            the host memory guard — and an out-of-band message when it stops an engine
<prefix>/<node-id>/transfers         downloads and mirror runs in flight
<prefix>/cluster                     head only: nodes, status, aggregate VRAM, version agreement
<prefix>/<node-id>/events/launch     a load finished: outcome, total, phase breakdown
<prefix>/<node-id>/logs/ainode       the orchestrator's own log (opt-in)
<prefix>/<node-id>/logs/vllm/<model> each engine instance's log (opt-in)
```

Per-model speed is tokens divided by the time spent generating them, not by
uptime — a model that served one request an hour ago is idle, not slow.

Log forwarding is off by default and sends only what is new since the last
publish, with progress bars dropped and a cap per message — a vLLM log is a
firehose. Every topic and every field is documented in
[docs/mqtt-schema.md](docs/mqtt-schema.md).

### Metrics — `/metrics` (Prometheus) and `/api/metrics` (JSON)

AINode exposes its own metrics on the same port as the API:

```bash
curl http://localhost:3000/metrics           # Prometheus text exposition
curl http://localhost:3000/api/metrics       # JSON snapshot
curl http://localhost:3000/api/metrics/gpu   # GPU subset
```

Key series:

- `ainode_uptime_seconds`, `ainode_build_info{version=...}`
- `ainode_requests_total`, `ainode_request_errors_total`
- `ainode_tokens_generated_total`, `ainode_tokens_per_second`
- `ainode_request_latency_milliseconds{quantile="0.5|0.95|0.99"}`
- `ainode_requests_by_model_total{model=...}`
- `ainode_gpu_utilization_percent`, `ainode_gpu_memory_used_bytes`, `ainode_gpu_temperature_celsius`

Scrape config for Prometheus:

```yaml
scrape_configs:
  - job_name: ainode
    static_configs:
      - targets: ["ainode-host:8000"]
```

---

## Requirements

- **OS**: Ubuntu 22.04+ (DGX Spark OS works out of the box)
- **GPU**: NVIDIA GB10 (DGX Spark, ASUS GX10) — or any NVIDIA GPU with
  CUDA 13 drivers
- **Memory**: 8 GB+ GPU for small models, 128 GB recommended for the
  big ones, 240 GB+ aggregated for real sharded work
- **Disk**: 20 GB for the container image; more for models
- **Docker**: 24.0+ with the NVIDIA Container Toolkit

---

## Why AINode?

| | Cloud AI | AINode |
|---|---|---|
| Monthly cost | $100–10,000+ | $0 (you own the hardware) |
| Data privacy | Your data on their servers | Your data stays local |
| Rate limits | Yes | None |
| Latency | 200–2000 ms | 10–50 ms |
| Fine-tuning | Limited, expensive | Unlimited, free |
| Internet required | Yes | No |
| Models available | Their choice | Your choice |

---

## Roadmap

- [x] Core CLI + installer
- [x] vLLM integration (patched NCCL for GB10)
- [x] Web UI (chat, server, downloads, training, config)
- [x] Multi-node auto-discovery + cluster topology view
- [x] Automatic model sharding across nodes (TP=2 verified)
- [x] NFS-shared model storage
- [x] Unified container image + systemd install
- [x] Browser-driven fine-tuning (LoRA / QLoRA / Full + DDP)
- [x] Training artifact retrieval, LoRA merge, checkpoint resume
- [x] Evaluation loop + W&B integration
- [x] Prometheus metrics endpoint (`/metrics`)
- [x] 4-node TP=4 sharded inference (verified — 235B-A22B-NVFP4)
- [x] In-browser quantization (AWQ / NVFP4) + push to Hugging Face
- [x] Federated multi-model serving (master routes `/v1/*` by model name)
- [x] Model stacking (N models per node, persisted + replayed)
- [x] Training + adapter-merge in a spawned GPU container (slim orchestrator)
- [x] Deploy pipeline (tag → CI → GHCR → `ainode update` / cluster update-all)
- [x] VLM (vision) serving with fp8-KV auto-skip on GB10
- [x] AutoData — Δ-filtered synthetic-data generation (val-set lift objective)
- [ ] Model marketplace (custom registries)
- [ ] Mobile-friendly UI

---

## Contributing

AINode is Apache-2.0 and welcomes contributions.

**For upstream AINode**, contribute at
[getainode/ainode](https://github.com/getainode/ainode) — that is the
maintained project, and most of what you might want to change lives there.

**For this fork**, work on a branch, open a PR, and run both before you do:

```bash
python -m pytest tests/ -q     # 2539 tests
ruff check ainode tests
```

Anything that only makes sense on a GB10 cluster belongs here; anything that
would work anywhere is better carried upstream, and
[`docs/fork-changes.md`](docs/fork-changes.md) marks which is which.

---

## License

Apache 2.0 — use it however you want.

---

<p align="center">
  <sub>crafted with <span style="color:#e74c3c">♥</span> by Jason Brashear · powered by <a href="https://argentos.ai">argentos.ai</a></sub>
</p>
