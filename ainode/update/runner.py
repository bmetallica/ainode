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
update uses. Which means the last thing this job does is end itself, and the
job file has to be written before that happens, not after.
"""

from __future__ import annotations

import logging
import os
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


class UpdateRunner:
    """One update at a time, with its output visible while it happens."""

    def __init__(self, app):
        self._app = app
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.job: dict = {"running": False, "status": "idle", "lines": [],
                          "started_at": 0.0, "finished_at": 0.0, "error": ""}

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
