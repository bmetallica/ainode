"""Suite-wide fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_engine_image_probe(monkeypatch):
    """A launch asks the engine image what it accepts (engine/image_probe.py)
    by starting a container. Nothing in the suite may do that: tests that
    exercise the probe or its use patch it themselves."""
    from ainode.engine.backends.eugr import EugrBackend

    monkeypatch.setattr(EugrBackend, "_engine_probe", lambda self: {})
    # Nor may a distributed launch in a test reach out to its "peers" to empty
    # their tuning cache (engine/autotune_cache.py) — those are test addresses.
    monkeypatch.setattr(EugrBackend, "_clear_autotune_cache", lambda self: None)
    yield
