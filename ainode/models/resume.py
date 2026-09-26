"""What is actually missing, before fetching it again.

Asked from the cluster, after a pull died on an unstable link:

    die hf modelle bestehen ja meist aus vielen einzeldateien. es wäre also
    schon gut wenn man einen button "Download fortsetzen" hätte welcher dann
    die vorhandenen dateien nach vollständigkeit prüft, evtl unvollständiges
    löscht und dann alles fehlende weiter herunterlädt

Three steps, and only the first is interesting. ``hf_hub_download`` already
skips a file it considers finished and already resumes one it does not, so
"download everything missing" needs no new machinery — a second pull IS that.
What it cannot do is notice a file that is the wrong size but whose metadata
says otherwise. A process killed mid-write, a filesystem that lost a tail, a
transfer that ended between the write and the rename: the local copy looks
accounted for and is short. Then the resume skips it and the engine finds out
later, which is the failure this whole area exists to prevent.

So: compare each local file against the size the Hub reports for it, remove the
ones that disagree along with the staging files, and let the ordinary download
path fetch what is now absent. Removing a wrong file is safe in a way that
keeping it is not — it costs one file of bandwidth and buys certainty.

Without the Hub (no route, no token, a gated repo) the size check cannot run.
It then reports that, removes only the staging files, and still resumes — an
answer of "I could not verify, and here is what I did anyway" is more useful
than a refusal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["hub_sizes", "inspect_local", "prepare_resume"]

#: Files that are ours rather than the repo's, and must never be judged or
#: removed by size.
_OURS = (".ainode-download-incomplete", ".ainode-backup")


def hub_sizes(hf_repo: str, token: Optional[str] = None) -> Dict[str, int]:
    """``{path: bytes}`` from the Hub, or {} when it cannot be reached."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(hf_repo, files_metadata=True)
    except Exception:
        logger.debug("could not read the file list for %s", hf_repo,
                     exc_info=True)
        return {}
    out: Dict[str, int] = {}
    for sibling in (getattr(info, "siblings", None) or []):
        name = getattr(sibling, "rfilename", "")
        size = getattr(sibling, "size", None)
        if size is None:
            lfs = getattr(sibling, "lfs", None)
            size = (lfs or {}).get("size") if isinstance(lfs, dict) else None
        if name and isinstance(size, int) and size > 0:
            out[name] = size
    return out


def _snapshot(directory: Path) -> Path:
    from ainode.planner.facts import snapshot_dir

    return snapshot_dir(directory) or directory


def inspect_local(directory, expected: Dict[str, int]) -> dict:
    """What is here, what is short, what is absent, what is staged.

    ``expected`` empty means the Hub could not be asked: everything present is
    then reported as unverified rather than as correct.
    """
    directory = Path(directory)
    snapshot = _snapshot(directory)
    staged = []
    try:
        staged = sorted(str(p.relative_to(directory))
                        for p in directory.rglob("*.incomplete"))
    except (OSError, ValueError):
        logger.debug("could not scan %s", directory, exc_info=True)

    present: List[str] = []
    wrong: List[dict] = []
    for name, size in sorted(expected.items()):
        path = snapshot / name
        if not path.is_file():
            continue
        try:
            actual = path.stat().st_size
        except OSError:
            continue
        if actual == size:
            present.append(name)
        else:
            wrong.append({"path": name, "on_disk": actual, "expected": size})
    missing = sorted(n for n in expected
                     if not (snapshot / n).is_file())
    return {
        "verified": bool(expected),
        "present": present,
        "wrong_size": wrong,
        "missing": missing,
        "staged": staged,
    }


def prepare_resume(directory, hf_repo: str,
                   token: Optional[str] = None) -> Tuple[dict, int]:
    """Clean the directory so an ordinary pull finishes it. (report, bytes freed).

    Removes the staging files and every local file whose size disagrees with
    the Hub. Does not fetch anything — that is the download path's job, and it
    already knows how.
    """
    directory = Path(directory)
    expected = hub_sizes(hf_repo, token)
    report = inspect_local(directory, expected)
    snapshot = _snapshot(directory)

    removed: List[str] = []
    freed = 0
    for entry in report["wrong_size"]:
        path = snapshot / entry["path"]
        try:
            freed += path.stat().st_size
            path.unlink()
            removed.append(entry["path"])
        except OSError:
            logger.debug("could not remove %s", path, exc_info=True)
    for relative in report["staged"]:
        path = directory / relative
        if any(part in relative for part in _OURS):
            continue
        try:
            freed += path.stat().st_size
            path.unlink()
            removed.append(relative)
        except OSError:
            logger.debug("could not remove %s", path, exc_info=True)

    report["removed"] = removed
    report["freed_bytes"] = freed
    # Anything removed is now missing, and the caller's pull will fetch it.
    report["will_fetch"] = sorted(set(report["missing"]) | {
        e["path"] for e in report["wrong_size"]})
    if not expected:
        report["note"] = (
            f"could not reach the Hub for {hf_repo}'s file list, so the files "
            f"already here were not size-checked. The staging files were "
            f"cleared and the download will carry on from what is on disk.")
    return report, freed
