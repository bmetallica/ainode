"""Is this checkpoint on disk actually all of it?

Reported from the cluster:

    wenn ainode (z.b. wegen einem update) neustartet während ein modell von
    hf heruntergeladen wird, dieses nach dem neustart als vollständig
    heruntergeladen gelistet wird … ich muss es erst komplett löschen und
    erneut herunterladen

The listing answered "is there a directory for this repo", which is a
different question and the wrong one. A download interrupted halfway leaves a
directory that looks exactly like a finished one — until an engine tries to
read a shard that is not there, minutes into a launch, and says something
about safetensors rather than about the download.

Two ways to be short, and both have to be checked because they leave
different traces:

* **mid-file.** huggingface_hub writes to ``<name>.incomplete`` and renames
  on success, so an interrupted transfer leaves that file behind — under
  ``.cache/huggingface/download/`` for a local_dir pull, next to the blob for
  a cache pull.
* **between files.** A shard that was never started leaves nothing at all.
  What catches that is the index the checkpoint ships with itself:
  ``model.safetensors.index.json`` lists every shard in its ``weight_map``,
  and a repo missing one of them is missing part of the model.

Neither check is clever and neither needs the network — which matters,
because this runs on a node that may have no route to the Hub, and the
question "have I got all of it" has to be answerable there.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["download_state", "is_complete", "INDEX_FILES"]

#: Index files that enumerate the shards a checkpoint is made of.
INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)


def _snapshot(directory: Path) -> Optional[Path]:
    """Where the files actually are, through the hub cache's layout."""
    from ainode.planner.facts import snapshot_dir

    resolved = snapshot_dir(directory)
    if resolved is not None:
        return resolved
    # A diffusion pipeline has model_index.json where a config.json would be,
    # so snapshot_dir does not recognise it.
    if (directory / "model_index.json").is_file():
        return directory
    snapshots = directory / "snapshots"
    if snapshots.is_dir():
        for sub in sorted(s for s in snapshots.iterdir() if s.is_dir()):
            return sub
    return directory if directory.is_dir() else None


def _partials(directory: Path) -> List[str]:
    """``.incomplete`` files left by an interrupted transfer."""
    try:
        return sorted(p.name for p in directory.rglob("*.incomplete"))
    except OSError:
        logger.debug("could not scan %s for partial files", directory,
                     exc_info=True)
        return []


def _shards_from_index(snapshot: Path) -> Iterable[str]:
    for name in INDEX_FILES:
        index = snapshot / name
        if not index.is_file():
            continue
        try:
            weight_map = json.loads(index.read_text()).get("weight_map") or {}
        except (OSError, ValueError):
            logger.debug("could not read %s", index, exc_info=True)
            continue
        yield from sorted(set(weight_map.values()))


def download_state(directory) -> Tuple[bool, str]:
    """(complete, reason). ``reason`` is "" when nothing is missing.

    Never raises and never answers "incomplete" on a doubt: a repo it cannot
    reason about — no index, no partial files — is reported complete, because
    refusing to launch a model that is fine is worse than the failure this
    catches.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return True, ""

    partial = _partials(directory)
    if partial:
        shown = ", ".join(p.replace(".incomplete", "") for p in partial[:3])
        more = f" and {len(partial) - 3} more" if len(partial) > 3 else ""
        return False, (
            f"a download of this model was interrupted: {len(partial)} file(s) "
            f"are still partial ({shown}{more}). Resume the download, or "
            f"delete the model and fetch it again.")

    snapshot = _snapshot(directory)
    if snapshot is None:
        return True, ""

    missing = [shard for shard in _shards_from_index(snapshot)
               if not (snapshot / shard).is_file()]
    if missing:
        shown = ", ".join(missing[:3])
        more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
        return False, (
            f"this model is missing {len(missing)} of the shards its own "
            f"index lists ({shown}{more}) — a download that was interrupted "
            f"between files leaves no other trace. Resume the download, or "
            f"delete the model and fetch it again.")
    return True, ""


def is_complete(directory) -> bool:
    return download_state(directory)[0]
