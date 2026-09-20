"""Log lines over MQTT: this node's own, and each engine instance's.

MQTT is a poor transport for a firehose and a vLLM log is one: it redraws a
progress bar several times a second and prints a line per weight shard. So
what is tested here is mostly restraint — that only NEW lines go out, that
the redraws do not, that a message is bounded and says when it dropped
something, and that none of it happens unless it was asked for.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ainode.telemetry.logs import (
    LogBuffer,
    LogPublisher,
    LogTail,
    MAX_BYTES,
    _slug,
)

APP_JS = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
          "static" / "js" / "app.js").read_text()


class _Config:
    node_id = "n1"
    node_name = "SPARK1"
    model = ""
    mqtt_logs = True
    mqtt_log_lines = 100
    mqtt_log_level = "INFO"


class _Record:
    def __init__(self, model):
        self.model = model


class _Backend:
    def __init__(self, path):
        self.log_path = path


class _Instance:
    def __init__(self, model, path):
        self.record = _Record(model)
        self.backend = _Backend(path)


class _Manager:
    def __init__(self, instances):
        self._instances = instances

    def instances(self):
        return list(self._instances)


class TestTailingAFile:
    def test_only_what_is_new_is_read(self, tmp_path):
        log = tmp_path / "vllm.log"
        log.write_text("first\nsecond\n")
        tail = LogTail(log)
        assert tail.read()[0] == ["first", "second"]
        assert tail.read()[0] == []
        with log.open("a") as handle:
            handle.write("third\n")
        assert tail.read()[0] == ["third"]

    def test_progress_bars_never_go_out(self, tmp_path):
        # A vLLM log is mostly this. Forwarding it would drown the broker and
        # bury the one line that matters.
        log = tmp_path / "vllm.log"
        log.write_text(
            "Capturing CUDA graphs (FULL):  48%|####  | 27/56 [00:04<00:03, 7.29it/s]\n"
            "Loading safetensors checkpoint shards:  40%|#### | 2/5\n"
            "ERROR: it went wrong\n")
        assert LogTail(log).read()[0] == ["ERROR: it went wrong"]

    def test_a_partial_line_waits_for_its_newline(self, tmp_path):
        # Publishing half a traceback line and the other half a second later
        # is worse than publishing it once, late.
        log = tmp_path / "vllm.log"
        log.write_text("complete\nhalf")
        tail = LogTail(log)
        assert tail.read()[0] == ["complete"]
        with log.open("a") as handle:
            handle.write(" a line\n")
        assert tail.read()[0] == ["half a line"]

    def test_a_relaunch_writes_a_new_file_and_is_followed(self, tmp_path):
        # A tail that kept seeking past the old end would go silent for the
        # rest of the node's uptime.
        log = tmp_path / "vllm.log"
        log.write_text("old run\n")
        tail = LogTail(log)
        tail.read()
        log.unlink()
        log.write_text("new run\n")
        assert tail.read()[0] == ["new run"]

    def test_truncation_in_place_is_followed_too(self, tmp_path):
        log = tmp_path / "vllm.log"
        log.write_text("a\nb\nc\nd\n")
        tail = LogTail(log)
        tail.read()
        log.write_text("fresh\n")
        assert tail.read()[0] == ["fresh"]

    def test_a_missing_file_is_silence_not_an_error(self, tmp_path):
        assert LogTail(tmp_path / "nope.log").read() == ([], 0)

    def test_starting_at_the_end_skips_the_history(self, tmp_path):
        # The first publish after a restart must not ship the whole of the
        # last load.
        log = tmp_path / "vllm.log"
        log.write_text("ancient\n" * 500)
        tail = LogTail(log)
        tail.start_at_end()
        assert tail.read()[0] == []


class TestBounds:
    def test_a_message_is_capped_and_says_so(self, tmp_path):
        log = tmp_path / "vllm.log"
        log.write_text("".join(f"line {i}\n" for i in range(500)))
        lines, dropped = LogTail(log).read(max_lines=10)
        assert len(lines) == 10
        assert dropped == 490
        # The newest are the ones kept.
        assert lines[-1] == "line 499"

    def test_bytes_are_capped_as_well(self, tmp_path):
        # One vLLM traceback line can be several kilobytes on its own.
        log = tmp_path / "vllm.log"
        log.write_text("".join("x" * 5000 + "\n" for _ in range(100)))
        tail = LogTail(log)
        lines, _ = tail.read(max_lines=100)
        assert sum(len(ln) + 1 for ln in lines) <= MAX_BYTES
        # And the gap is reported rather than passed off as the whole log.
        assert tail.skipped_bytes > 0

    def test_lines_dropped_by_the_count_cap_are_reported(self, tmp_path):
        log = tmp_path / "vllm.log"
        log.write_text("short\n" * 300)
        assert LogTail(log).read(max_lines=10)[1] == 290


class TestThisNodesOwnLog:
    def test_it_captures_what_ainode_logs(self):
        buffer = LogBuffer()
        record = logging.LogRecord("ainode.test", logging.INFO, __file__, 1,
                                   "something happened", None, None)
        buffer.emit(record)
        lines, _ = buffer.drain()
        assert "something happened" in lines[0]
        assert "INFO" in lines[0]

    def test_draining_empties_it(self):
        buffer = LogBuffer()
        buffer.emit(logging.LogRecord("a", logging.INFO, __file__, 1, "x",
                                      None, None))
        assert buffer.drain()[0]
        assert buffer.drain()[0] == []

    def test_the_level_filters(self):
        buffer = LogBuffer()
        for level in (logging.DEBUG, logging.INFO, logging.ERROR):
            buffer.emit(logging.LogRecord("a", level, __file__, 1, f"l{level}",
                                          None, None))
        lines, _ = buffer.drain(min_level=logging.ERROR)
        assert len(lines) == 1 and "ERROR" in lines[0]

    def test_a_handler_never_raises(self):
        # A logging handler that throws takes down whatever was logging.
        buffer = LogBuffer()
        buffer.setFormatter(None)
        buffer.emit(logging.LogRecord("a", logging.INFO, __file__, 1,
                                      "%d", ("not a number",), None))


class TestThePublisher:
    def _app(self, tmp_path, models=("org/one",)):
        instances = []
        for i, model in enumerate(models):
            path = tmp_path / f"vllm-{i}.log"
            path.write_text("")
            instances.append(_Instance(model, path))
        return {"config": _Config(), "instances": _Manager(instances)}, instances

    def test_nothing_is_published_unless_it_was_asked_for(self, tmp_path):
        app, _ = self._app(tmp_path)
        app["config"].mqtt_logs = False
        assert LogPublisher(app).payloads() == {}

    def test_one_topic_per_instance(self, tmp_path):
        app, instances = self._app(tmp_path, ("org/one", "org/two"))
        publisher = LogPublisher(app)
        publisher.payloads()                      # first pass registers tails
        for instance in instances:
            with Path(instance.backend.log_path).open("a") as handle:
                handle.write("a line\n")
        payloads = publisher.payloads()
        assert "logs/vllm/org_one" in payloads
        assert "logs/vllm/org_two" in payloads
        assert payloads["logs/vllm/org_one"]["lines"] == ["a line"]

    def test_the_payload_names_the_node_and_the_instance(self, tmp_path):
        app, instances = self._app(tmp_path)
        publisher = LogPublisher(app)
        publisher.payloads()
        with Path(instances[0].backend.log_path).open("a") as handle:
            handle.write("hello\n")
        payload = publisher.payloads()["logs/vllm/org_one"]
        assert payload["node_id"] == "n1"
        assert payload["instance"] == "org_one"
        assert payload["source"] == "vllm"
        assert payload["count"] == 1

    def test_a_quiet_instance_publishes_nothing(self, tmp_path):
        # An empty message every interval per instance is how a broker fills
        # up with nothing.
        app, _ = self._app(tmp_path)
        publisher = LogPublisher(app)
        publisher.payloads()
        assert publisher.payloads() == {}

    def test_the_first_pass_does_not_ship_the_backlog(self, tmp_path):
        app, instances = self._app(tmp_path)
        Path(instances[0].backend.log_path).write_text("old\n" * 100)
        assert LogPublisher(app).payloads() == {}

    def test_a_model_id_becomes_a_safe_topic_level(self):
        # Slashes would make new levels; + and # are wildcards.
        assert _slug("org/model-1.5B") == "org_model-1.5B"
        assert "/" not in _slug("a/b/c")
        assert _slug("") == "instance"


class TestTheEngineLogsAreSeparateFiles:
    def test_each_instance_writes_its_own(self):
        from unittest import mock

        from ainode.core.config import NodeConfig
        from ainode.engine.backends import eugr

        with mock.patch.object(eugr, "detect_gpu", return_value=None):
            primary = eugr.EugrBackend(NodeConfig(node_id="n", model="a/b"))
            stacked = eugr.EugrBackend(NodeConfig(node_id="n", model="c/d"),
                                       instance_id="8001")
        # The primary keeps the plain name: documentation and muscle memory
        # point at it.
        assert primary.log_path.name == "vllm.log"
        assert stacked.log_path.name == "vllm-8001.log"
        assert primary.log_path != stacked.log_path


class TestTheSettings:
    def test_the_switch_is_off_by_default(self):
        from ainode.core.config import NodeConfig

        assert NodeConfig(node_id="n").mqtt_logs is False

    def test_the_settings_are_exposed_and_saved(self):
        import asyncio
        import json

        from ainode.telemetry.api_routes import _settings, handle_put_settings

        class _Cfg(_Config):
            mqtt_enabled = False
            mqtt_host = "broker"
            saved = False

            def save(self):
                type(self).saved = True

        assert _settings(_Cfg())["mqtt_logs"] is True

        class _Req:
            def __init__(self, app, body):
                self.app = app
                self._body = body

            async def json(self):
                return self._body

        app = {"config": _Cfg()}
        resp = asyncio.run(handle_put_settings(_Req(app, {
            "mqtt_logs": True, "mqtt_log_lines": 25, "mqtt_log_level": "warning"})))
        body = json.loads(resp.body)
        assert body["settings"]["mqtt_log_lines"] == 25
        assert body["settings"]["mqtt_log_level"] == "WARNING"

    def test_an_unknown_level_is_refused(self):
        import asyncio

        class _Cfg(_Config):
            mqtt_enabled = False
            mqtt_host = "broker"

            def save(self):
                pass

        class _Req:
            def __init__(self, app, body):
                self.app = app
                self._body = body

            async def json(self):
                return self._body

        from ainode.telemetry.api_routes import handle_put_settings

        resp = asyncio.run(handle_put_settings(
            _Req({"config": _Cfg()}, {"mqtt_log_level": "CHATTY"})))
        assert resp.status == 400

    def test_the_topics_are_listed_when_it_is_on(self):
        from ainode.telemetry.api_routes import _topic_examples

        class _Cfg(_Config):
            mqtt_topic_prefix = "ainode"

        topics = _topic_examples(_Cfg())
        assert "ainode/n1/logs/ainode" in topics
        assert "ainode/n1/logs/vllm/+" in topics

    def test_the_ui_offers_the_switch(self):
        assert "cfg-mqtt-logs" in APP_JS
        assert "mqtt_log_level" in APP_JS
        assert "firehose" in APP_JS
