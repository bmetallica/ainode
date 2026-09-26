"""A dropped packet is not a failed download.

Reported from the cluster, on a link that comes and goes:

    der download (unvollständig) wird übrigens immernoch in der modelliste als
    fertig (ohne hinweiß das er nicht vollständig ist) gelistet ... kannst du
    noch irgendwas an der downloadstabilität machen damit er nicht bei kleinen
    netzaussetzern komplett abbricht sondern einfach weiter macht?

The pull fetches files through a bounded thread pool and propagates the FIRST
error, cancelling every file not yet started. One timeout on one shard of
twenty-four therefore ends the whole transfer, twenty gigabytes in. Nothing
about that is necessary: ``hf_hub_download`` resumes from the ``.incomplete``
file it left behind, so retrying a file costs the lost chunk and not the file.

What must NOT be retried is as important as what must. A 401 will be a 401
again, a 404 will be a 404 again, and a full disk will still be full — retrying
those turns a clear error into a slow one. So the rule is a deny-list of things
that will not change, and everything else gets another go.
"""

from __future__ import annotations

import errno
import logging
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["should_retry", "retry_delay", "MAX_ATTEMPTS"]

#: Attempts per file, the first included. Five tries over about a minute of
#: backoff covers a link that drops for a few seconds at a time; beyond that
#: the link is down rather than flaky, and the job should say so instead of
#: sitting there.
MAX_ATTEMPTS = 5

#: HTTP statuses that will give the same answer next time.
_FINAL_STATUS = (401, 403, 404, 416)

#: Errno values where retrying makes things worse, not better.
_FINAL_ERRNO = (errno.ENOSPC, errno.EACCES, errno.EPERM, errno.EROFS,
                errno.EDQUOT if hasattr(errno, "EDQUOT") else errno.ENOSPC)


def _status_of(exc: BaseException) -> Optional[int]:
    """The HTTP status behind an exception, if it carries one."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def should_retry(exc: BaseException) -> bool:
    """Is this worth another attempt?

    Never for a cancellation — that is the operator's decision and repeating it
    would override them.
    """
    name = type(exc).__name__
    if "Cancel" in name:
        return False
    if isinstance(exc, (KeyboardInterrupt, SystemExit, MemoryError)):
        return False
    status = _status_of(exc)
    if status in _FINAL_STATUS:
        return False
    if isinstance(exc, OSError) and exc.errno in _FINAL_ERRNO:
        return False
    # Anything else: a reset connection, a read timeout, a 502 from a CDN edge,
    # an incomplete chunked read. All of them are "try again".
    return True


def retry_delay(attempt: int) -> float:
    """Seconds to wait before attempt ``attempt`` (1-based, so 2 is the first
    retry). Doubling from two seconds, capped — long enough for a radio to
    reacquire, short enough that a whole repo does not stall on one file."""
    return min(30.0, 2.0 * (2 ** max(0, attempt - 2)))
