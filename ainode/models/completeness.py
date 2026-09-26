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

The two are not equal, and the order they are asked in matters. The index is
*proof*: it is the checkpoint's own account of what it is made of, and a
directory holding every shard it names is complete. A staging file is only
*evidence*, and a Xet transfer leaves it under a content id rather than a
filename, so it can outlive the file it was staging without anything noticing.
So the index is asked first, and the staging files only get to answer for a
repo that ships no index at all. What made that concrete: a model carried in
by hand through ``/model-import``, complete, reported as an interrupted
download because of two stubs left by the transfer it had replaced — and sent
its owner to "delete the model and fetch it again".
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["download_state", "is_complete", "clear_partials",
           "mark_download_started", "clear_download_mark",
           "DOWNLOAD_MARK", "INDEX_FILES"]

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


#: Written when a download starts, removed when it finishes. The only signal
#: here that is a statement rather than an inference.
DOWNLOAD_MARK = ".ainode-download-incomplete"


def mark_download_started(directory, repo: str = "", **extra) -> None:
    """Record that a download of ``repo`` into ``directory`` has begun.

    Every other check in this module infers completeness from what the
    directory looks like, and a directory can look finished while being half
    of one. Measured on the cluster: a pull over an unstable link died on the
    fourth of twenty-four shards, and the listing showed the model as On disk
    with no hint — because the shard index had not been fetched yet (so there
    was nothing to check the shards against), the tokenizer files that HAD
    arrived satisfied the tokenizer check, and the in-flight files finished
    renaming themselves before anybody looked. Three heuristics, all agreeing
    on the wrong answer.

    This is not a heuristic. Something wrote down that it was starting and did
    not write down that it finished.
    """
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"repo": repo, "started_at": time.time()}
        payload.update({k: v for k, v in extra.items() if v is not None})
        (directory / DOWNLOAD_MARK).write_text(json.dumps(payload, indent=2))
    except OSError:
        logger.debug("could not mark %s as downloading", directory,
                     exc_info=True)


def clear_download_mark(directory) -> bool:
    """Remove the mark. True when one was there."""
    path = Path(directory) / DOWNLOAD_MARK
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.debug("could not clear the download mark in %s", directory,
                     exc_info=True)
        return False


def _download_mark(directory: Path) -> str:
    """"" unless a download into ``directory`` is on record as unfinished."""
    path = directory / DOWNLOAD_MARK
    if not path.is_file():
        return ""
    detail = ""
    try:
        blob = json.loads(path.read_text())
        error = str(blob.get("last_error") or "")
        attempts = blob.get("attempts")
        if error:
            detail = f" The last attempt stopped with: {error}"
        elif attempts:
            detail = f" {attempts} attempt(s) were made."
    except (OSError, ValueError):
        pass
    return (
        "a download of this model started and did not finish — that is on "
        "record, not inferred from the files." + detail + " Resume it, or "
        "delete the model and fetch it again. If you completed it by hand, "
        "importing it clears this."
    )


def _partial_paths(directory: Path) -> List[Path]:
    try:
        return sorted(directory.rglob("*.incomplete"))
    except OSError:
        logger.debug("could not scan %s for partial files", directory,
                     exc_info=True)
        return []


def _partials(directory: Path) -> List[str]:
    """``.incomplete`` files left by an interrupted transfer, relative.

    Relative rather than by name, because the name is not always a name. A
    Xet-backed transfer writes its staging files under the file's content id,
    so the basename is a base64 hash and says nothing:

        0k4AjklGyGCyIWbHx36RsIjxBNg=.3565b5c5....f12932e7

    The path around it is what makes that legible — it sits under
    ``.cache/huggingface/download/``, which is the download cache and not the
    checkpoint.
    """
    out = []
    for path in _partial_paths(directory):
        try:
            out.append(str(path.relative_to(directory)))
        except ValueError:          # pragma: no cover - rglob cannot do this
            out.append(path.name)
    return out


def clear_partials(directory) -> Tuple[int, int]:
    """Delete the ``.incomplete`` staging files under ``directory``.

    (files removed, bytes reclaimed). For the import path, which is an
    operator saying "these are the files": whatever an earlier interrupted
    download left staged is then both stale and, at Xet chunk sizes, tens of
    gigabytes of it. Never raises — a stub that cannot be removed is a stub
    that stays, which is the situation before the call.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return 0, 0
    files = 0
    total = 0
    if clear_download_mark(directory):
        files += 1
    for path in _partial_paths(directory):
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            logger.debug("could not remove %s", path, exc_info=True)
            continue
        files += 1
        total += size
    return files, total


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


#: Any one of these is a tokenizer a transformers model can be built from.
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json",
                    "spiece.model", "vocab.txt")


def _tokenizer_gap(snapshot: Path) -> str:
    """"" unless the tokenizer files are visibly a partial set.

    A model is more than its weights, and the shard index says nothing about
    the rest of the repo. Measured on the cluster, after a download that
    reported itself finished::

        added_tokens.json   1660
        merges.txt          2414077

    and nothing else — no vocab.json, no tokenizer.json, no
    tokenizer_config.json. merges.txt is half of a BPE pair; on its own it
    cannot build anything. The engine got as far as the weights and then:

        ValueError: Couldn't instantiate the backend tokenizer from one of:

    Narrow on purpose. A diffusion pipeline keeps its tokenizer in a
    subdirectory and is skipped; a repo with any one of the files above is
    left alone, because plenty ship exactly one.
    """
    if not (snapshot / "config.json").is_file():
        return ""          # not a transformers-style model
    if not any(snapshot.glob("*.safetensors")) and \
            not any(snapshot.glob("*.bin")):
        return ""          # no weights: not a model directory at all
    # The more specific reading first: merges.txt is half of a BPE pair, and
    # saying so is more useful than "no tokenizer", which is also true.
    if (snapshot / "merges.txt").is_file() \
            and not (snapshot / "vocab.json").is_file() \
            and not (snapshot / "tokenizer.json").is_file():
        return (
            "this model has merges.txt and neither vocab.json nor "
            "tokenizer.json — half of a BPE tokenizer, which cannot build "
            "anything. The download stopped partway through the small files "
            "at the end of the repo. Resume it, or delete the model and "
            "fetch it again.")
    if not any((snapshot / name).is_file() for name in _TOKENIZER_FILES):
        return (
            "this model has weights but no tokenizer file at all — none of "
            + ", ".join(_TOKENIZER_FILES) + ". The download stopped before "
            "the small files at the end of the repo. Resume it, or delete "
            "the model and fetch it again.")
    return ""


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

    # Before any inference: did something say it was downloading and never say
    # it had finished. The checks below read the files and can all three be
    # satisfied by a tree that is missing most of itself.
    marked = _download_mark(directory)
    if marked:
        return False, marked

    snapshot = _snapshot(directory)
    if snapshot is not None:
        tokenizer = _tokenizer_gap(snapshot)
        if tokenizer:
            return False, tokenizer

        shards = list(_shards_from_index(snapshot))
        missing = [shard for shard in shards
                   if not (snapshot / shard).is_file()]
        if missing:
            shown = ", ".join(missing[:3])
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            return False, (
                f"this model is missing {len(missing)} of the shards its own "
                f"index lists ({shown}{more}) — a download that was "
                f"interrupted between files leaves no other trace. Resume the "
                f"download, or delete the model and fetch it again.")
        if shards:
            # Proof beats evidence. The index is the checkpoint's own account
            # of what it is made of, every shard it names is here, and the
            # tokenizer is here — so the model is complete, whatever staging
            # files an earlier attempt left lying around. Measured on the
            # cluster, after a model was brought in through /model-import over
            # the top of an interrupted Xet download:
            #
            #   2 file(s) are still partial (0k4AjklGyGCyIWbHx36RsIjxBNg=.…)
            #
            # Every shard was present. The stubs were from the transfer that
            # the import replaced, and refusing to launch over them sent the
            # operator to "delete the model and fetch it again" — which would
            # have destroyed the files they had just carried in by hand.
            return True, ""

    partial = _partials(directory)
    if partial:
        # No index to appeal to, so the staging files are the only evidence
        # there is, and they do decide.
        shown = ", ".join(p.replace(".incomplete", "") for p in partial[:3])
        more = f" and {len(partial) - 3} more" if len(partial) > 3 else ""
        return False, (
            f"a download of this model was interrupted: {len(partial)} file(s) "
            f"are still partial ({shown}{more}). Resume the download, or "
            f"delete the model and fetch it again.")
    return True, ""


def is_complete(directory) -> bool:
    return download_state(directory)[0]
