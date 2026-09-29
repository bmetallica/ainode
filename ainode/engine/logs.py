"""Reading an engine log, which is not a text file in the strict sense.

An engine writes progress bars, ANSI colour and carriage returns, and a
container that is killed mid-write leaves a truncated multi-byte character
behind. ``Path.read_text()`` on that raises UnicodeDecodeError — a
ValueError, not an OSError — so a reader guarding only against OSError loses
the whole log, silently, and the UI shows an empty box exactly when there is
something to read.

Observed from the operator's side first::

    $ grep -n "recipe environment" ~/.ainode/logs/vllm.log
    grep: /home/admin/.ainode/logs/vllm.log: binary file matches

grep says the same thing in its own way, and answers with nothing.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["read_log_tail", "rotate_log"]

#: Read at most this much from the end. A log grows across every launch a
#: node has ever done; the tail is what anyone wants, and reading 400 MB to
#: return 100 lines is a cost paid on every poll of the UI.
_TAIL_BYTES = 2 * 1024 * 1024


def read_log_tail(path, lines: int = 100) -> str:
    """The last ``lines`` of a log, whatever bytes it happens to contain.

    Undecodable bytes are replaced rather than raising: a log is evidence,
    and evidence with one broken character in it is still evidence.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > _TAIL_BYTES:
                handle.seek(size - _TAIL_BYTES)
                handle.readline()      # drop the half line the seek landed in
            raw = handle.read()
    except OSError:
        logger.debug("could not read %s", path, exc_info=True)
        return ""
    text = raw.decode("utf-8", errors="replace")
    # Carriage returns are how a progress bar redraws itself. Splitting on
    # them as well turns one 4000-character line into the frames it was,
    # which is what makes a tail of N lines mean anything on an engine log.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(text.splitlines()[-lines:])


#: Rotate a log at the start of a launch once it has grown past this, and
#: keep this many old ones beside it (name.log.1 … name.log.5). Every launch
#: appended to the same file forever: 35 MB of distributed.log on the head
#: after a few weeks, and a grep over the logs — which is what every how-to
#: here says to do — getting slower with each launch.
ROTATE_BYTES = 50 * 1024 * 1024
ROTATE_KEEP = 5


def rotate_log(path, max_bytes: int = ROTATE_BYTES,
               keep: int = ROTATE_KEEP) -> bool:
    """Move ``path`` to ``path.1`` (and so on) if it is larger than
    ``max_bytes``. Called before a launch opens its log, so the current file
    always begins with the launch it belongs to. Never raises."""
    path = Path(path)
    try:
        if path.stat().st_size < max_bytes:
            return False
    except OSError:
        return False
    try:
        oldest = path.with_name(f"{path.name}.{keep}")
        if oldest.exists():
            oldest.unlink()
        for index in range(keep - 1, 0, -1):
            older = path.with_name(f"{path.name}.{index}")
            if older.exists():
                older.rename(path.with_name(f"{path.name}.{index + 1}"))
        path.rename(path.with_name(f"{path.name}.1"))
        return True
    except OSError:
        logger.debug("could not rotate %s", path, exc_info=True)
        return False
