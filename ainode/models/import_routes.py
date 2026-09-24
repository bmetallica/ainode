"""Bring a model in by hand, when the link cannot carry it.

    kann ich die modelle eigentlich auch irgendwie auf einem anderen pc
    downloaden und dann nach ainode bringen (also z.b. mit einem usbstick)?
    ich vermute das die LTE leitung hier am abbrechenden download schuld ist.

An 86 GB checkpoint over LTE is a bet against the connection staying up for
hours, and this deployment has now lost that bet twice — once losing the
tokenizer files, once the shards. A download that has to be perfect to be
useful is the wrong shape for a link that is not.

So: three endpoints, and nothing clever.

``GET  /api/models/import/plan``   what this repo needs, what is already
                                  here, and the exact URL for each missing
                                  file. Asks the Hub when it can reach it and
                                  falls back to the checkpoint's own index
                                  when it cannot — the second is what a node
                                  with no route has, and it is enough to
                                  finish a partial download.
``POST /api/models/import/upload`` one file, streamed to its place under the
                                  model directory. Streamed, because these
                                  are gigabytes and a buffered read is the
                                  node's memory.
``POST /api/models/import/finish`` re-check completeness and push the result
                                  to the peers, the same way a download does.

The rule the rest of the product follows holds here too: everything comes
through the head, and the peers get their copies from it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

from aiohttp import web

from ainode.api.params import as_object, str_field

logger = logging.getLogger(__name__)

__all__ = ["register_import_routes"]

#: A path inside the repo. No absolute paths, no "..", no backslashes — this
#: names a file that will be created under the model directory, and it
#: arrives from a browser.
_SAFE_RELATIVE = re.compile(r"^(?!/)(?!.*\.\.)[A-Za-z0-9._\-/]+$")

#: Files a transformers checkpoint needs beyond its weights. Used only when
#: the Hub cannot be reached: it is a floor, not an inventory.
_COMPANIONS = (
    "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "merges.txt", "added_tokens.json",
    "special_tokens_map.json", "preprocessor_config.json",
    "chat_template.jinja",
)


def register_import_routes(app: web.Application) -> None:
    app.router.add_get("/api/models/import/plan", handle_plan)
    app.router.add_post("/api/models/import/upload", handle_upload)
    app.router.add_post("/api/models/import/finish", handle_finish)


def _target_dir(app, hf_repo: str) -> Path:
    manager = app["model_manager"]
    # The flat layout the downloader writes, so an imported model sits where
    # a downloaded one would and every other code path finds it unchanged.
    return Path(manager.models_dir) / hf_repo.replace("/", "--")


def _hub_files(hf_repo: str) -> List[dict]:
    """[{path, size}] from the Hub, or [] when it cannot be reached."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(hf_repo, files_metadata=True)
    except Exception:
        logger.debug("could not list %s on the Hub", hf_repo, exc_info=True)
        return []
    out = []
    for sibling in (getattr(info, "siblings", None) or []):
        name = getattr(sibling, "rfilename", "")
        if not name:
            continue
        size = getattr(sibling, "size", None)
        if not size:
            lfs = getattr(sibling, "lfs", None)
            size = (lfs or {}).get("size", 0) if isinstance(lfs, dict) else 0
        out.append({"path": name, "size": int(size or 0)})
    return out


def _local_expectation(directory: Path) -> List[dict]:
    """What the files on disk say is still missing, with no network.

    The shard index names every weight file; the companions are the small
    ones that a download loses last and that nothing else notices gone.
    """
    from ainode.models.completeness import INDEX_FILES
    import json as _json

    wanted: List[str] = list(_COMPANIONS)
    for name in INDEX_FILES:
        index = directory / name
        if not index.is_file():
            continue
        wanted.append(name)
        try:
            weight_map = _json.loads(index.read_text()).get("weight_map") or {}
        except (OSError, ValueError):
            continue
        wanted.extend(sorted(set(weight_map.values())))
    seen = set()
    out = []
    for path in wanted:
        if path in seen:
            continue
        seen.add(path)
        out.append({"path": path, "size": 0})
    return out


async def handle_plan(request: web.Request) -> web.Response:
    """GET /api/models/import/plan?hf_repo= — what to fetch, and from where."""
    hf_repo = (request.query.get("hf_repo") or "").strip()
    if not hf_repo or "/" not in hf_repo:
        return web.json_response({"error": "hf_repo required"}, status=400)

    directory = _target_dir(request.app, hf_repo)
    listing = _hub_files(hf_repo)
    source = "hub"
    if not listing:
        listing = _local_expectation(directory)
        source = "local"

    files = []
    missing_bytes = 0
    for entry in listing:
        path = entry["path"]
        local = directory / path
        present = local.is_file()
        size = int(entry.get("size") or 0)
        if present:
            try:
                size = local.stat().st_size or size
            except OSError:
                pass
        elif source == "hub":
            missing_bytes += size
        files.append({
            "path": path,
            "size": size,
            "present": present,
            # The plain resolve URL, which is what a browser on another
            # machine can be pointed at.
            "url": f"https://huggingface.co/{hf_repo}/resolve/main/{path}",
        })

    complete, reason = (True, "")
    if directory.is_dir():
        from ainode.models.completeness import download_state

        complete, reason = download_state(directory)

    return web.json_response({
        "hf_repo": hf_repo,
        "target_dir": str(directory),
        # Which list this is. A local one cannot know about files the
        # checkpoint's own index does not mention, and saying so is the
        # difference between a plan and a guess.
        "source": source,
        "files": files,
        "missing": [f for f in files if not f["present"]],
        "missing_bytes": missing_bytes,
        "complete": complete,
        "incomplete_reason": reason,
    })


async def handle_upload(request: web.Request) -> web.Response:
    """POST /api/models/import/upload — one file, streamed into place.

    multipart/form-data with ``hf_repo``, ``path`` (the file's path inside
    the repo) and ``file``. Streamed to a temporary name and renamed on
    success, so an interrupted upload never leaves a file that looks whole.
    """
    try:
        reader = await request.multipart()
    except Exception:
        return web.json_response({"error": "expected multipart/form-data"},
                                 status=400)

    hf_repo = ""
    rel_path = ""
    written = 0
    target: Optional[Path] = None

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "hf_repo":
            hf_repo = (await part.text()).strip()
            continue
        if part.name == "path":
            rel_path = (await part.text()).strip()
            continue
        if part.name != "file":
            await part.read()
            continue

        if not hf_repo or "/" not in hf_repo:
            return web.json_response({"error": "hf_repo must come before the "
                                               "file part"}, status=400)
        rel_path = rel_path or (part.filename or "")
        if not rel_path or not _SAFE_RELATIVE.match(rel_path):
            return web.json_response(
                {"error": f"{rel_path!r} is not a path inside a repo"},
                status=400)

        target = _target_dir(request.app, hf_repo) / rel_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".ainode-upload")
            with open(tmp, "wb") as sink:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    sink.write(chunk)
                    written += len(chunk)
            tmp.replace(target)
        except OSError as exc:
            return web.json_response(
                {"error": f"could not write {target}: {exc}"}, status=500)

    if target is None:
        return web.json_response({"error": "no file part"}, status=400)

    _forget_size(request.app, target.parent)
    logger.info("imported %s (%d bytes) into %s", rel_path, written,
                target.parent)
    return web.json_response({"ok": True, "hf_repo": hf_repo,
                              "path": rel_path, "bytes": written})


async def handle_finish(request: web.Request) -> web.Response:
    """POST /api/models/import/finish {hf_repo} — check it, then spread it."""
    try:
        body = as_object(await request.json()) if request.can_read_body else {}
    except Exception:
        body = {}
    hf_repo = str_field(body, "hf_repo", "model").strip()
    if not hf_repo or "/" not in hf_repo:
        return web.json_response({"error": "hf_repo required"}, status=400)

    directory = _target_dir(request.app, hf_repo)
    if not directory.is_dir():
        return web.json_response(
            {"error": f"nothing imported for {hf_repo} yet"}, status=404)

    from ainode.models.completeness import download_state

    complete, reason = download_state(directory)
    result = {"ok": True, "hf_repo": hf_repo, "complete": complete,
              "incomplete_reason": reason}
    if not complete:
        # Not mirrored: sending an incomplete model to the peers spreads the
        # problem rather than the model.
        result["mirrored"] = False
        return web.json_response(result)

    job: dict = {}
    from ainode.models.api_routes import _mirror_after_download

    await _mirror_after_download(request.app, hf_repo, job)
    result["mirrored"] = True
    result["mirror"] = job.get("mirror", {})
    return web.json_response(result)


def _forget_size(app, path) -> None:
    manager = app.get("model_manager")
    forget = getattr(manager, "forget_size", None)
    if callable(forget):
        try:
            forget(path)
        except Exception:
            logger.debug("could not drop the cached size", exc_info=True)
