"""Applying a profile as a job the dashboard can follow.

Phase 7 of wizzard.md. Applying a profile takes minutes — every model has to
answer before the next one on its node starts — and the request used to stay
open for all of it, with nothing to show until the end and nothing to do if a
model failed halfway. As a job:

* every entry has a state the page polls: pending, starting, loading, ready,
  unchanged, slow, failed, skipped;
* it can be **cancelled** — between two entries, never in the middle of one;
* what ran before is **captured first**, so a profile that turns out wrong can
  be undone with one click (a second job applying that capture);
* at the end, what each entry **actually took** is gathered from every node it
  ran on and written into the profile, where the wizard shows it beside what
  was planned.

One job at a time: two applies converging the same nodes onto different
profiles would stop each other's models.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional

from ainode.profiles.apply import ApplyProgress, _entry_key, apply_profile, capture_profile
from ainode.profiles.store import Profile, ProfileEntry

logger = logging.getLogger(__name__)

__all__ = ["ApplyJob", "start_apply_job", "current_job", "get_job"]

#: How long to wait for the measurements of the last model to be written.
#: The recorder runs every five seconds and records a load when its phase
#: turns ready; a little longer than that covers the peers' announcements.
MEASURE_WAIT = 20.0

#: Jobs kept for the page to read back after they finish.
KEEP = 10

_TASKS: set = set()

#: One start at a time per app: the capture before an apply awaits the peers,
#: and a second click in that window must not start a second job.
_STARTING: Dict[int, asyncio.Lock] = {}


class ApplyJob(ApplyProgress):
    def __init__(self, profile: Profile, before: Optional[Profile] = None,
                 restore_of: str = ""):
        self.id = uuid.uuid4().hex[:12]
        self.profile = profile
        self.before = before
        self.restore_of = restore_of
        self.state = "running"
        self.phase_text = "Starting"
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.cancel_requested = False
        self.entries: Dict[str, dict] = {}
        self.order: List[str] = []
        self.report: Optional[dict] = None
        self.measured: Dict[str, dict] = {}
        self.error = ""

    # -- ApplyProgress ---------------------------------------------------------

    def phase(self, text: str) -> None:
        self.phase_text = text

    def entry(self, entry: ProfileEntry, state: str, detail: str = "") -> None:
        key = _entry_key(entry)
        if key not in self.entries:
            self.order.append(key)
            self.entries[key] = {"model": entry.model, "kind": entry.kind,
                                 "node_ids": list(entry.node_ids or [])}
        row = self.entries[key]
        if row.get("state") != state:
            row["since"] = time.time()
        row.update(state=state, detail=detail)

    def cancelled(self) -> bool:
        return self.cancel_requested

    # -- reading ---------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id, "profile": self.profile.name, "state": self.state,
            "phase": self.phase_text, "started_at": self.started_at,
            "finished_at": self.finished_at, "cancel_requested": self.cancel_requested,
            "entries": [dict(self.entries[k], key=k) for k in self.order],
            "stopped": list((self.report or {}).get("stopped") or []),
            "unreachable": list((self.report or {}).get("unreachable") or []),
            "ok": bool(self.report and self.report.get("ok")),
            "can_restore": self.before is not None and self.state != "running",
            "restore_of": self.restore_of, "measured": self.measured,
            "error": self.error, "now": time.time(),
        }


def _jobs(app) -> Dict[str, ApplyJob]:
    jobs = app.get("profile_jobs")
    if jobs is None:
        jobs = {}
        app["profile_jobs"] = jobs
    return jobs


def get_job(app, job_id: str) -> Optional[ApplyJob]:
    return _jobs(app).get(job_id)


def current_job(app) -> Optional[ApplyJob]:
    """The running job, or the most recent one."""
    jobs = list(_jobs(app).values())
    running = [j for j in jobs if j.state == "running"]
    if running:
        return running[0]
    return max(jobs, key=lambda j: j.started_at) if jobs else None


async def start_apply_job(app, profile: Profile, *, capture_before: bool = True,
                          restore_of: str = "") -> ApplyJob:
    """Start applying ``profile`` in the background. Raises RuntimeError if a
    job is already running."""
    lock = _STARTING.setdefault(id(app), asyncio.Lock())
    if lock.locked():
        raise RuntimeError("A profile is already being started.")
    async with lock:
        return await _start(app, profile, capture_before, restore_of)


async def _start(app, profile: Profile, capture_before: bool, restore_of: str) -> ApplyJob:
    running = [j for j in _jobs(app).values() if j.state == "running"]
    if running:
        raise RuntimeError(f"Profile {running[0].profile.name!r} is still being applied.")
    before = None
    if capture_before:
        try:
            # The capture asks every peer how it started its models — over
            # the network, synchronously — so not on the event loop.
            before = await asyncio.get_event_loop().run_in_executor(
                None, capture_profile, app, "previous-state",
                f"What ran before {profile.name} was applied")
        except Exception:
            logger.exception("could not capture the state before applying %s",
                             profile.name)
    job = ApplyJob(profile, before=before, restore_of=restore_of)
    jobs = _jobs(app)
    jobs[job.id] = job
    for old in sorted(jobs.values(), key=lambda j: j.started_at)[:-KEEP]:
        if old.state != "running":
            jobs.pop(old.id, None)

    async def _run() -> None:
        try:
            job.report = await apply_profile(app, profile, wait=True, progress=job)
            if job.cancel_requested:
                job.state = "cancelled"
            else:
                job.state = "done" if job.report.get("ok") else "failed"
            job.phase_text = "Measuring what it took"
            await _record_measurements(app, job)
            job.phase_text = {"done": "Done", "failed": "Done, with errors",
                              "cancelled": "Cancelled"}[job.state]
        except asyncio.CancelledError:
            job.state = "cancelled"
            raise
        except Exception as exc:
            logger.exception("applying %s failed", profile.name)
            job.state = "failed"
            job.error = str(exc)
            job.phase_text = "Failed"
        finally:
            job.finished_at = time.time()

    # On the app's loop, not the request's: the request returns at once.
    task = asyncio.get_event_loop().create_task(_run())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job


async def _record_measurements(app, job: ApplyJob) -> None:
    """Gather what each entry took, from every node, into the profile."""
    wanted = [e for e in job.profile.entries
              if (job.entries.get(_entry_key(e)) or {}).get("state") in ("ready", "slow")]
    if not wanted:
        return
    from ainode.measure.api_routes import gather_cluster_measurements

    deadline = time.time() + MEASURE_WAIT
    found: Dict[str, dict] = {}
    while True:
        try:
            by_model = await gather_cluster_measurements(app)
        except Exception:
            logger.debug("could not gather measurements", exc_info=True)
            by_model = {}
        for entry in wanted:
            key = _entry_key(entry)
            if key in found:
                continue
            for row in by_model.get(entry.model) or []:
                # Only a measurement of THIS launch, on a node this entry uses
                # (a distributed one is recorded by the head it launched from).
                fresh = float(row.get("last_ok") or 0) >= job.started_at
                here = not entry.node_ids or row.get("node_id") in entry.node_ids
                if fresh and here and row.get("memory_gb"):
                    found[key] = {k: row.get(k) for k in (
                        "memory_gb", "memory_by_node", "weights_gb", "kv_cache_gb",
                        "kv_tokens", "max_model_len", "gpu_memory_utilization",
                        "load_seconds", "engine_version", "node_id")}
                    found[key]["at"] = row.get("last_ok")
                    break
        if len(found) == len(wanted) or time.time() >= deadline:
            break
        await asyncio.sleep(5)
    job.measured = found
    if not found:
        return
    store = app.get("profiles")
    if store is None:
        return
    saved = store.get(job.profile.name)
    if saved is None:
        return
    measured = dict(saved.measured or {})
    measured.update(found)
    saved.measured = measured
    try:
        store.put(saved)
    except Exception:
        logger.exception("could not write the measurements into %s", saved.name)
