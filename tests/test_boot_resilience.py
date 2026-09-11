"""A node must stay reachable when its engine cannot start.

The case that motivated this: a head configured across three Sparks with one
of them powered off. `start_distributed()` raises when a peer refuses SSH, the
exception escaped `cmd_start` before `run_server`, and systemd's
`Restart=always` turned that into a crash loop every 10 s — no UI to see the
reason, and no way to relaunch across the nodes that were up.

These drive `cmd_start` with a stubbed engine and assert only what matters:
the web server still runs, and it runs without an engine rather than with a
half-started one.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from ainode.cli import main as cli_main
from ainode.core.config import NodeConfig


class _Engine:
    """Engine stub whose start() outcome the test chooses."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.stopped = False

    def start(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def stop(self):
        self.stopped = True


def _boot(engine, **config_overrides):
    """Run cmd_start far enough to reach run_server; return its kwargs."""
    config = NodeConfig(
        node_id="head", model="m", engine_strategy="docker",
        onboarded=True, **config_overrides,
    )
    captured = {}

    def _run_server(**kwargs):
        captured.update(kwargs)

    # get_backend and run_server are imported inside cmd_start, so both have to
    # be patched where they are defined, not on the cli module.
    with mock.patch("ainode.engine.backends.get_backend", return_value=engine), \
         mock.patch.object(cli_main.NodeConfig, "load", classmethod(lambda cls: config)), \
         mock.patch.object(cli_main, "ensure_dirs", lambda *a, **k: None), \
         mock.patch.object(cli_main, "_remove_pid", lambda *a, **k: None), \
         mock.patch("ainode.models.api_routes.consume_start_clean", lambda: False), \
         mock.patch("ainode.core.gpu.detect_gpu", return_value=None), \
         mock.patch("ainode.api.server.run_server", _run_server):
        cli_main.cmd_start(SimpleNamespace(
            port=None, api_port=None, model=None, in_container=True,
        ))
    return captured


class TestEngineFailureDoesNotTakeTheNodeDown:
    def test_raising_engine_still_serves_the_ui(self):
        """The crash-loop case: an unreachable peer raises out of start()."""
        engine = _Engine(RuntimeError(
            "ssh docker run -d for worker on 10.0.0.13 failed (rc=255)"
        ))
        captured = _boot(engine, distributed_mode="head",
                         peer_ips=["10.0.0.12", "10.0.0.13"])
        assert captured, "run_server was never reached"
        assert captured["engine"] is None

    def test_engine_returning_false_still_serves_the_ui(self):
        captured = _boot(_Engine(False))
        assert captured["engine"] is None

    def test_successful_engine_is_passed_through(self):
        engine = _Engine(True)
        captured = _boot(engine)
        assert captured["engine"] is engine

    def test_no_systemexit_on_a_failed_distributed_head(self):
        """sys.exit(1) here is what systemd turns into a restart loop."""
        engine = _Engine(RuntimeError("peer unreachable"))
        try:
            _boot(engine, distributed_mode="head", peer_ips=["10.0.0.12"])
        except SystemExit:  # pragma: no cover — the regression this guards
            pytest.fail("cmd_start exited instead of serving the UI")

    def test_shutdown_does_not_trip_over_a_missing_engine(self):
        """The finally block calls engine.stop(); engine is None by then."""
        captured = _boot(_Engine(False))
        assert captured["engine"] is None  # reached the end without raising
