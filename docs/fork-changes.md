# What this fork changes

Upstream is [getainode/ainode](https://github.com/getainode/ainode), Apache-2.0,
and so is this. Everything AINode does — the container, the web UI, the
OpenAI-compatible proxy, discovery, the vLLM launch path, training — is theirs.
This document is the inventory of what has been added on top, and why each
piece exists.

It is written for two readers: someone deciding whether to use this fork or
upstream, and someone deciding whether a piece of it is worth carrying back.
Section 6 says which is which.

Measured against `upstream/main` @ a98987d.

---

## 1. Why the fork exists

Three DGX Sparks wired in a **switchless ring** — each ConnectX-7 port a
private cable to a different neighbour, with a shared 10G Ethernet for
coordination. Two assumptions in the original break on that topology:

1. **One cluster NIC per node.** `cluster_interface` is a single field that
   names the NCCL socket interface, the announced address, the SSH/Ray target
   and the subnet filter for HCA selection all at once. That holds while every
   node shares one ConnectX-7 subnet. In a ring it does not: no CX7 subnet
   reaches all three nodes.
2. **Tensor parallelism only.** The launch path always built TP = node count.
   TP splits attention heads and head counts are powers of two, so **TP=3 has
   no models behind it** — three nodes were unusable.

Everything else here grew out of running that cluster daily.

---

## 2. What was added

New packages, none of which exist upstream:

| Package | What it does | Why |
|---|---|---|
| `ainode/cluster/topology.py` | Classifies the fabric by active CX7 link count (2 = direct-attach, 4 = mesh) and derives which interface carries coordination and which devices carry RDMA | The reason for the fork |
| `ainode/engine/parallelism.py` | `Strategy`, `ParallelPlan`, `plan_for`, `validate_plan` — tensor, pipeline and data axes | TP=3 does not exist; pipeline does |
| `ainode/engine/distribute.py`, `acquire.py`, `mirror.py` | Weights and engine images move head → peers over the fabric; a sub-node fetches from a neighbour rather than from the internet | Measured: 80 MB/s from a peer against 1.1 MB/s from Hugging Face |
| `ainode/engine/load_phase.py` | Reads the engine's own output into a coarse phase, a detail line, a failure cause **and a per-phase clock** | A five-minute load showed "starting · 8%" throughout, which is indistinguishable from a hang |
| `ainode/engine/serve_args.py` | Merges vLLM flags per flag rather than wholesale | Typing one advanced field used to drop the rest of a model's recipe |
| `ainode/profiles/` | "What the cluster should be serving", captured from what is running, applied to converge — each entry on the node it names, and a copy running anywhere else stopped — one of them the default at boot | Bringing three nodes back after a restart was a dozen manual steps |
| `ainode/placement/` | One model, one node set, remembered | A cluster settles into an arrangement; re-picking it on every relaunch is re-configuring rather than running |
| `ainode/planner/` | Reads the checkpoint's own `config.json` and the nodes' free memory and computes: does it fit, which axis, what `max-model-len`, how much KV, how many concurrent users — and shows its working | The alternative was a person with a calculator, once per model |
| `ainode/safety/` | Host memory guard, the admission gate in front of both launch paths, the cap that keeps `gpu_memory_utilization` inside what is free, and the refusal built from what the guard has already had to kill | Two nodes were lost to a launch that did not fit — and again to a model card's 0.97, which on unified memory is 124 GB of 128 |
| `ainode/measure/` | Writes down what each launch actually cost and prefers a measurement to an estimate | Every number already existed; nobody was recording it |
| `ainode/assist/` | Explains a failure using a model that is already running | A vLLM traceback assumes you already know this hardware |
| `ainode/telemetry/` | MQTT: system, GPU, fabric, models, engine, safety, transfers, cluster, logs, events | A scrape needs the monitor to reach every node; a publish needs each node to reach one broker |
| `ainode/metrics/system.py`, `metrics/fabric.py` | Host sampling, and RoCE counters read from `/sys/class/infiniband` | `/proc/net/dev` reports zero while RDMA saturates the fabric |
| `ainode/clients/opencode.py` | Generates a client config from what is actually serving | Three settings per model that are wrong in ways that surface hours later |
| `ainode/models/tool_parsers.py` | Picks a tool-call and reasoning parser from the model family | |
| `ainode/api/params.py` | One place that coerces request fields | |
| `ainode/update/` | Update from the fork's source: is the branch ahead of the commit this image was built from, and the `git pull` + `update-cluster.sh` run behind a button — including its own restart, since the run ends by stopping the container it runs in | Upstream compares a version against the latest published tag; this deployment builds from a checkout, where the code moves far more often than the version |
| `ainode/engine/backends/diffusers.py`, `engine/diffusers_server.py` | A second engine kind: image generation | vLLM cannot load a diffusers pipeline at all |

Upstream's `ainode/bench/` was extended rather than replaced: selectable
context size and padded prompts, so a throughput number says at what length it
was measured.

### The model search

Worth its own note, because it decides what an operator can find at all.
Upstream searches `text-generation` and judges a repo runnable by looking for
`mlx`, `gguf` or `ggml` in its name. Here the search knows **kinds** — chat,
vision, image generation, embeddings — and a kind is not a label: it decides
which engine runs the model and therefore which formats can work. So:

* the Hub is queried per kind, and the filter in the UI changes the *query*
  rather than narrowing the results of the wrong one;
* embedding models are findable (upstream's filter excluded them, correctly
  for the vLLM engine and wrongly for a deployment that serves them
  in-process);
* the servability verdict is per kind, read from the Hub's tags and library
  name rather than the repo title, and **carries its reason**. GGUF on a chat
  model is llama.cpp's format; GGUF on an image model holds the transformer
  alone while the text encoder is the larger half; Nunchaku is not
  "unsupported" but has no aarch64 build of its kernels. That last distinction
  is the one that tells someone whether to wait.

Any quantisation that can actually be loaded is offered — fp8, int4, AWQ,
GPTQ, NVFP4, bf16. The line is not the bit width but the layout.

---

## 3. What was changed in upstream's own files

The large ones, by how much:

| File | Roughly | What changed |
|---|---|---|
| `web/static/js/app.js` | +2500 | Everything above that has a screen: the launch planner's hint, the instance card and its details dialog, profiles, placement, the memory-guard settings, the image panel, the cluster graphic's node detail |
| `api/server.py` | +970 | ~120 routes where upstream has ~29; the cluster projections, the proxy's image route, the sync loop's extra duties |
| `engine/backends/eugr.py` | +930 | Per-instance containers, scripts and logs; the parallel axes; image and weight distribution; the phase tracker; `kill()` |
| `models/registry.py` | +740 | Exact repo sizes, search by kind and its per-kind format verdict, the curated cluster catalog, modality, the size cache |
| `models/api_routes.py` | +680 | Stacked instances, the admission gate, per-load overrides that reset rather than leak, engine detection from the checkpoint |
| `engine/sharding_routes.py` | +465 | The real parallel plan, remembered placement, node-failure relaunch, the refusal messages |
| `scripts/install.sh` | +60 | The `/ainode-src` mount so the UI can update from the checkout, and keeping the image already installed when the registry is unreachable |
| `service/systemd.py` | +34 | The same mount, in the renderer the CLI uses |

`scripts/` gained the base-image build pinned to an eugr commit, the
orchestrator image, the image-generation engine, the registry cache, the
cluster updater and a diagnostic.

Tests: **88 new files**, 13 upstream files extended, 2987 tests. Two of them
hold the documentation to the code rather than to good intentions:
`test_fork_documentation.py` checks that every module and link this file and
the README name exists and that every feature row claiming an endpoint matches
the live router, and `test_mqtt_schema_doc.py` checks that every published
field appears in the schema. Change a published field and one of them fails.

---

## 4. What was not changed

Worth saying, because it bounds the surface: the OpenAI-compatible request and
response shapes, the training subsystem, the onboarding flow, the discovery
protocol's wire format (only fields were added, and every one of them is
optional so an older peer still parses), and the installer's contract — one
idempotent command that pins an image, writes a unit and keeps your config. A
client pointed at this fork sees upstream's API plus routes it can ignore.

The two unit renderers are the exception to that last one and are listed in
section 3: both gained the source mount, and the installer will keep an
already-installed image rather than fail when the registry cannot be reached.
What it promises is the same; what it writes into `ExecStart` is one mount
longer.

---

## 5. The hardware facts this fork encodes

These are the things that cost days to learn and are written into the code and
its comments rather than kept in someone's head:

* **The GPU's memory is the host's memory.** On GB10 there is no separate
  VRAM. An engine that over-allocates does not get a CUDA error — it starves
  the kernel and the node needs a power cycle. Hence the memory guard, and
  hence `enable_model_cpu_offload()` being explicitly *not* used: "offloading"
  moves bytes within one pool and pays for the copies.
* **`nvidia-smi` reports `[N/A]` for memory here**, and NVML returns `used=0`
  rather than raising. A metric that reads zero is worse than one that is
  absent, so the collector falls back to the host figure.
* **RDMA bypasses the kernel network stack**, so `/proc/net/dev` — and
  therefore psutil, and therefore every ordinary network metric — shows zero
  while the fabric is saturated. The real counters are in
  `/sys/class/infiniband/<hca>/ports/<n>/counters/`, and `port_xmit_data`
  counts **32-bit words, not bytes**.
* **Decode is memory-bandwidth bound**, so a dense model is slow and an MoE of
  the same size is fast, and adding nodes does not make a single stream
  faster.
* **A stale compiled-kernel cache produces a CUDA fault** with nothing
  pointing at the cache, after an engine image changes.
* **TP must be 2, 4 or 8**; pipeline needs the architecture to implement it;
  three nodes have no tensor split.

---

## 6. What is worth carrying upstream

Not mesh-specific, and useful on any cluster:

* the **load phases and their timings** — every deployment has the "is it
  stuck or working" question;
* the **planner**, minus its GB10 constants;
* **profiles** and **placement**;
* the **error assistant**;
* the **measurement store** — it makes a catalog self-correcting;
* the **per-instance containers, scripts and logs**, without which two models
  loading at once interleave into one file;
* two plain bug fixes found along the way: a head that crash-looped when a
  peer was offline, and a completed download reporting stale progress because
  its poller raced the completion write.

Specific to this hardware and probably not wanted elsewhere: the fabric
topology detection, the RoCE counters, the host memory guard's defaults, and
the curated model catalog.

---

## 7. Attribution

The network heuristics and the multi-node launch path are adapted from
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT) —
`autodiscover.sh`, `launch-cluster.sh` and `docs/NETWORKING.md` document this
hardware. Borrowed logic is marked at each site; there are no verbatim copies.
`scripts/_eugr/` is a shallow checkout at a pinned commit, read-only.

AINode itself is [Jason Brashear's](https://jasonbrashear.com). The UI footer
keeps that credit and adds *forked by bmetallica*, linking here, so a running
instance — or a screenshot of one — says which code it is.

## 8. Further reading

| | |
|---|---|
| [`docs/mqtt-schema.md`](mqtt-schema.md) | Every topic and field published over MQTT |
| [`update.md`](../update.md) | The audit that produced the memory guard, the non-blocking launches and the UI pass |
| [`images.md`](../images.md) | The image-generation plan, and what was built from it |
| [`docs/mesh/ANLEITUNG-3-NODE-MESH.md`](mesh/ANLEITUNG-3-NODE-MESH.md) | Step-by-step for the reference cluster (German) |
| [`docs/mesh/PHASE1-ANALYSE.md`](mesh/PHASE1-ANALYSE.md) | The audit of the original code before any of it was touched (German) |
| [`CHANGELOG.md`](../CHANGELOG.md) | Everything, in order |
