# Bootstrapping this fork's image

This repo publishes and installs its **own** container image — nothing points at
`getainode` or `ainode.dev` any more. That means the very first install needs an
image to exist under this owner. There are two ways to get one, and after that
the normal `install.sh` path works unchanged.

## Option A — publish from CI (what releases use)

Requires a self-hosted runner labelled `[self-hosted, dgx-spark, aarch64]` with
docker + nvidia-container-toolkit, and the runner user in the `docker` group.

Start it from the browser (repository → **Actions** → *publish-image* → **Run
workflow**) or from the CLI — `gh` is a convenience, not a requirement:

```bash
gh workflow run publish-image.yml -f push=false   # build only, proves the runner works
gh workflow run publish-image.yml -f push=true    # build and push
```

A `v*` tag push (e.g. `v0.5.7`) builds and pushes automatically.

The GHCR namespace follows `github.repository_owner`, so there is nothing to
edit when the fork moves. Docker Hub mirroring stays off unless the
`DOCKERHUB_ORG` repository *variable* and the `DOCKERHUB_USER` /
`DOCKERHUB_TOKEN` secrets are all set.

**Make the package public** after the first push, or every node needs a pull
secret: GitHub → Packages → `ainode` → Package settings → Change visibility.
`install.sh` resolves tags anonymously and expects a public image.

## First: switch CI on

GitHub disables workflows on a **forked** repository until someone enables them
once by hand. The repo-level API reports `enabled: true` and each workflow reads
`state: active` while that gate is still closed, so the absence is easy to miss —
this fork merged its first four pull requests with nothing running.

Open **Actions** in the repository and click *"I understand my workflows, go
ahead and enable them"*. There is no API for it. Confirm afterwards:

```bash
gh run list --repo <owner>/ainode --limit 5
```

An empty list on a repository that has had pushes means the gate is still shut.
`tests.yml` then runs `ruff` and `pytest` on every pull request and on pushes to
`main`; `publish-image.yml` stays manual / tag-triggered.

## Option B — build locally on one Spark (no GitHub account needed)

No CI, no registry, no account — `git clone` of a public repository and the
public base it pulls need neither. Clone first: the build runs out of the
working tree.

```bash
git clone https://github.com/bmetallica/ainode && cd ainode
scripts/build-base-image.sh                            # also clones eugr's repo
docker build -f scripts/Dockerfile.ainode -t ainode:dev .
AINODE_IMAGE=ainode:dev bash scripts/install.sh
```

`install.sh` checks for a locally present image before reaching for a registry,
so `ainode:dev` installs without a `docker pull` that could only fail — it would
resolve to `docker.io/library/ainode`.

`AINODE_IMAGE` is written to `~/.ainode/image.env`, which the systemd unit reads,
so the node keeps booting that image across restarts.

To put the same image on the other nodes, `docker save` it and `docker load` on
each — or point them at a registry you control with
`AINODE_GHCR_REPO=<registry>/<owner>/ainode`.

## NCCL version floor — required for the mesh

`scripts/build-base-image.sh` pins the NCCL that goes into the base image.
**`NCCL_IB_SUBNET_AWARE_ROUTING` exists only from NCCL `v2.30.7-1` onward**
(`src/transport/net_ib/connect.cc`, `NCCL_PARAM(IbSubnetAwareRouting, …)`); it
is absent from 2.28.x, 2.29.x and 2.30.3.

That parameter is what makes a switchless ring work — each CX7 port reaches a
different neighbour, so NCCL has to choose the HCA whose subnet reaches the
peer. Below the floor AINode sets the variable, NCCL ignores it, and the
failure looks like a hang rather than a misconfiguration.

The default pin is `v2.30.7-1`. The parameter defaults to `0`, so a switched
cluster behaves exactly as it did on the previous 2.28.3 pin — this is a floor,
not a mesh-only build. Override if you need to:

```bash
AINODE_NCCL_TAG=v2.31.2-1 scripts/build-base-image.sh
```

`ainode doctor` reports the running NCCL version and warns when a mesh is
detected below the floor.

> Upstream's published `ghcr.io/getainode/ainode-base:*` images were built
> before this pin and carry NCCL 2.28.3 — usable on a switched cluster, **not**
> on a mesh. A mesh needs a base image built here.

## If the base build fails on a dependency conflict

`build-and-copy.sh` fetches its vLLM wheel from eugr's **rolling**
`prebuilt-vllm-current` release, so the pinned `EUGR_COMMIT` does not pin the
wheel. A bad nightly there fails the build with something like:

```
Because quack-kernels==0.6.4 depends on nvidia-cutlass-dsl==4.6.2
and vllm==… depends on nvidia-cutlass-dsl[cu13]==4.7.0, …
we can conclude that your requirements are unsatisfiable.
```

That is upstream's wheel, not this repo. Options, in order of preference:

1. **Wait and retry.** The tag is republished regularly; a broken closure is
   usually corrected within a day.
2. **Reuse a cached wheel set.** `build-and-copy.sh` keeps previously
   downloaded wheels; a build that succeeded before can be repeated without a
   fresh download. Check `~/.cache` for the wheel dir the script prints.
3. **Force the resolution** with a uv override, accepting that the combination
   is untested:
   ```bash
   docker build -f scripts/Dockerfile.ainode ... \
     --build-arg UV_OVERRIDE="nvidia-cutlass-dsl==4.7.0"
   ```
   Only do this if you are prepared to verify the resulting engine actually
   generates tokens — a forced pin can produce an image that loads and then
   fails in a kernel.

## Where the registry is configured

One constant per language, not a string scattered across files:

| Place | What it sets |
|---|---|
| `ainode/core/config.py` → `AINODE_GHCR_REPO` | the running node's update checks and the rendered systemd unit; override with `$AINODE_GHCR_REPO` |
| `scripts/install.sh` → `AINODE_GHCR_REPO` | tag resolution, the pull, and the `ainode` wrapper it installs |
| `.github/workflows/publish-image.yml` → `REGISTRY_GHCR` | derived from the repository owner; no edit needed |

`AINODE_PROJECT_URL` (same module) is the link in the systemd unit and the
installer hints.
