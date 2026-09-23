"""Running the update: git pull, then the cluster script.

Two things make this different from the image-pull update that already
exists, and both shape what is here.

**The checkout lives on the host.** AINode runs in a container that mounts
``~/.ainode``, the docker socket and the SSH keys — but not the repository,
because until now nothing in the container needed it. So the update either
finds the checkout mounted or it cannot run, and when it cannot the useful
thing is to say exactly which mount is missing rather than to fail at `git`.

**The script restarts this node last.** It does that deliberately: peers
first, head last, so a failure leaves the head able to report it. Inside a
container the head's restart is a ``docker stop`` of ourselves, which systemd
turns into a start of the new image — the same trick the existing in-container
update uses. Which means the last thing this job does is end itself, so the
outcome is written to disk as it happens: after the restart this object is a
fresh one with an empty job, and the UI would otherwise show an update that
apparently never ran.

**The container needs tools the orchestrator image was built without.** It is
a slim Python image; ``git`` was added for exactly this feature. An older
image on the node cannot run the update that would replace it, so the
preflight says so in those words instead of letting subprocess raise
``No such file or directory: 'git'``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["UpdateRunner", "DEFAULT_SOURCE_DIR", "CONTAINER_SOURCE_DIR"]

#: Where the installer puts the checkout on the host.
DEFAULT_SOURCE_DIR = "/opt/ainode"

#: Where the unit mounts it into the container.
CONTAINER_SOURCE_DIR = "/ainode-src"

#: Tail kept for the UI. A build is thousands of lines and the last few
#: hundred are the ones anyone reads.
_MAX_LINES = 400

#: What the update shells out to, and what each one is for. Checked before
#: anything starts: the run ends with this node restarting, so a failure
#: halfway leaves a cluster on two versions.
_TOOLS = {
    "git": "pull the branch",
    "docker": "build and distribute the image",
    "ssh": "reach the other nodes",
}


def _last_run_path(app) -> Path:
    from ainode.core.config import AINODE_HOME

    return AINODE_HOME / "update-last.json"


class UpdateRunner:
    """One update at a time, with its output visible while it happens."""

    def __init__(self, app):
        self._app = app
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.job: dict = {"running": False, "status": "idle", "lines": [],
                          "started_at": 0.0, "finished_at": 0.0, "error": ""}
        self._restore()

    def _restore(self) -> None:
        """The last run, from disk.

        A successful update ends by stopping this container, so the process
        that reports the result is never the process that ran it.
        """
        try:
            saved = json.loads(_last_run_path(self._app).read_text())
        except Exception:
            return
        if isinstance(saved, dict) and saved.get("status"):
            saved["running"] = False
            saved["restored"] = True
            self.job = saved

    def _persist(self) -> None:
        path = _last_run_path(self._app)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.job))
            tmp.replace(path)
        except Exception:
            logger.debug("could not write %s", path, exc_info=True)

    def missing_tools(self, *, nodes: List[str]) -> List[str]:
        """Which of the tools the update needs are not on PATH."""
        needed = [t for t in _TOOLS if t != "ssh" or nodes]
        return [tool for tool in needed if shutil.which(tool) is None]

    # -- where the source is ------------------------------------------------

    def source_dir(self) -> Optional[Path]:
        """The checkout as THIS process can see it, or None.

        In a container that is the mount; on a host install it is the
        directory itself. Both are checked, because a development install
        runs outside a container and would otherwise be told to add a mount
        it does not need.
        """
        configured = str(getattr(self._app.get("config"), "source_dir", "")
                         or "").strip()
        candidates = [CONTAINER_SOURCE_DIR, configured or DEFAULT_SOURCE_DIR]
        if configured:
            candidates.append(configured)
        for candidate in candidates:
            path = Path(candidate)
            if (path / ".git").exists() and (path / "scripts" /
                                             "update-cluster.sh").is_file():
                return path
        return None

    def why_not(self) -> str:
        """"" when an update can run, else what is missing and what to do."""
        if self.source_dir() is not None:
            return ""
        host_dir = str(getattr(self._app.get("config"), "source_dir", "")
                       or "").strip() or DEFAULT_SOURCE_DIR
        if os.environ.get("AINODE_IN_CONTAINER"):
            return (
                f"The checkout is not visible from inside the container. "
                f"AINode mounts ~/.ainode, the docker socket and the SSH keys, "
                f"but not the source — so a source update needs one more "
                f"mount. Add it by re-running the installer ON THE HOST, "
                f"which is idempotent and keeps your config.json:\n\n"
                f"    curl -fsSL https://raw.githubusercontent.com/"
                f"bmetallica/ainode/main/scripts/install.sh | bash\n\n"
                f"It mounts {host_dir} at {CONTAINER_SOURCE_DIR} when a "
                f"checkout is there, and the button works from then on. "
                f"(`ainode service install` cannot do it: on the host that "
                f"command is a wrapper into this container, and the unit is "
                f"written by the installer.) Until then, update from the "
                f"head's shell: scripts/update-cluster.sh")
        return (f"No checkout at {host_dir} — set the right path in "
                f"Settings → Updates.")

    def last_summary(self) -> Optional[dict]:
        """A one-line account of the last run, for the panel's Status card."""
        job = self.job
        if not job.get("status") or job.get("status") == "idle":
            return None
        return {
            "status": job.get("status", ""),
            "running": bool(job.get("running")),
            "finished_at": job.get("finished_at", 0.0),
            "error": job.get("error", ""),
            "restored": bool(job.get("restored")),
            "nodes": list(job.get("nodes") or []),
        }

    def tool_message(self, missing: List[str]) -> str:
        """Why this image cannot update itself, and what does it once.

        The orchestrator image is deliberately slim and had no git until the
        feature that needs it existed. A node on an older image therefore
        cannot run the update that would give it one — which is a bootstrap
        problem, not a broken button, and the difference is the whole content
        of this message.
        """
        what = ", ".join(f"{tool} (to {_TOOLS[tool]})" for tool in missing)
        if not os.environ.get("AINODE_IN_CONTAINER"):
            return f"This node is missing: {what}. Install it and try again."
        return (
            f"This AINode image is missing: {what}. The orchestrator image is "
            f"a slim Python container and only carries git from the release "
            f"that introduced updating from source — so an older image cannot "
            f"run the update that would replace it. Do this once, from the "
            f"head's shell:\n\n"
            f"    cd {DEFAULT_SOURCE_DIR} && git pull\n"
            f"    scripts/update-cluster.sh --nodes <peer1>,<peer2>\n\n"
            f"That builds and distributes an image that has it, and the button "
            f"works from then on.")

    # -- running ------------------------------------------------------------

    def start(self, *, nodes: List[str], base: bool = False,
              images: bool = False) -> dict:
        with self._lock:
            if self.job.get("running"):
                return {"ok": False, "status": 409,
                        "error": "an update is already running"}
            blocked = self.why_not()
            if blocked:
                return {"ok": False, "status": 409, "error": blocked}
            missing = self.missing_tools(nodes=nodes)
            if missing:
                return {"ok": False, "status": 409,
                        "error": self.tool_message(missing)}
            self.job = {"running": True, "status": "running", "lines": [],
                        "started_at": time.time(), "finished_at": 0.0,
                        "error": "", "nodes": list(nodes)}
        self._thread = threading.Thread(
            target=self._run, args=(list(nodes), base, images),
            name="ainode-update", daemon=True)
        self._thread.start()
        return {"ok": True}

    def _say(self, line: str) -> None:
        self.job["lines"].append(line.rstrip())
        del self.job["lines"][:-_MAX_LINES]

    def _run(self, nodes: List[str], base: bool, images: bool) -> None:
        directory = self.source_dir()
        try:
            self._say(f"[ainode] source: {directory}")
            self._step(["git", "pull", "--ff-only"], directory)
            command = ["scripts/update-cluster.sh"]
            if nodes:
                command += ["--nodes", ",".join(nodes)]
            if base:
                command.append("--base")
            if images:
                command.append("--images")
            self._step(command, directory)
            self.job["status"] = "done"
        except Exception as exc:
            self.job["status"] = "failed"
            self.job["error"] = str(exc)
            self._say(f"[ainode] FAILED: {exc}")
        finally:
            self.job["running"] = False
            self.job["finished_at"] = time.time()
            # Before the script's last step can end this process.
            self._persist()

    def _step(self, command: List[str], cwd: Path) -> None:
        self._say(f"[ainode] $ {' '.join(command)}")
        process = subprocess.Popen(
            command, cwd=str(cwd), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
        for line in process.stdout or []:
            self._say(line)
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"{command[0]} exited with {code}")
