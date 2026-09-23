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

__all__ = ["read_log_tail"]

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
