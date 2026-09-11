"""Total coercions for JSON request bodies.

A handler that writes ``(body.get("model") or "").strip()`` assumes two things
the client controls: that the body decoded to a JSON *object*, and that the
field holds a *string*. Neither survives contact with ``[1, 2]`` or
``{"model": 123}`` — ``.get`` and ``.strip`` raise ``AttributeError``, aiohttp
turns that into a 500 with a traceback, and a plainly malformed request reads
like a server fault in the logs.

These helpers make the cast total. A wrong type collapses to the empty /
default value, so the handler's own ``if not model: return 400`` fires and the
caller gets the right status with the right message. Nothing here validates
*meaning* — that stays with the handler, which knows what a valid model id or
node set looks like.
"""

from __future__ import annotations

from typing import Any, List, Optional

__all__ = ["as_object", "int_field", "str_field", "str_list_field"]


def as_object(body: Any) -> dict:
    """Return ``body`` if it decoded to a JSON object, else an empty dict.

    ``await request.json()`` yields whatever the client sent — a list, a bare
    string and ``null`` are all valid JSON. Callers that immediately reach for
    ``.get`` need this first.
    """
    return body if isinstance(body, dict) else {}


def str_field(body: Any, *names: str, default: str = "") -> str:
    """First of ``names`` present as a string, stripped; ``default`` otherwise.

    Several endpoints accept a field under more than one name (``hf_repo`` or
    ``model_id``), hence the varargs. A present-but-non-string value is treated
    as absent rather than coerced with ``str()``: ``str({"a": 1})`` would
    produce a plausible-looking id that fails much later, somewhere less
    informative.
    """
    obj = as_object(body)
    for name in names:
        value = obj.get(name)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return default


def str_list_field(body: Any, name: str) -> List[str]:
    """``name`` as a list of non-empty strings; ``[]`` for anything else.

    A bare string is **not** accepted as a one-element list: ``node_ids:
    "head"`` would otherwise iterate into ``['h', 'e', 'a', 'd']`` and produce
    an error naming four nodes that were never asked for. Non-string entries
    inside a real list are dropped.
    """
    value = as_object(body).get(name)
    if not isinstance(value, list):
        return []
    return [v.strip() for v in value if isinstance(v, str) and v.strip()]


def int_field(
    body: Any,
    name: str,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    """``name`` as an int, clamped to ``[minimum, maximum]``; ``default`` if absent.

    Accepts an int or a numeric string, since form-ish clients send ``"2"``.
    Booleans are rejected: ``True`` is an ``int`` in Python, and a caller who
    sent ``true`` meant a flag, not 1.
    """
    value = as_object(body).get(name)
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed
