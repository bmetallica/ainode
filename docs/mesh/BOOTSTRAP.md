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

## Where the registry is configured

One constant per language, not a string scattered across files:

| Place | What it sets |
|---|---|
| `ainode/core/config.py` → `AINODE_GHCR_REPO` | the running node's update checks and the rendered systemd unit; override with `$AINODE_GHCR_REPO` |
| `scripts/install.sh` → `AINODE_GHCR_REPO` | tag resolution, the pull, and the `ainode` wrapper it installs |
| `.github/workflows/publish-image.yml` → `REGISTRY_GHCR` | derived from the repository owner; no edit needed |

`AINODE_PROJECT_URL` (same module) is the link in the systemd unit and the
installer hints.
