"""The MQTT schema document has to describe the MQTT AINode actually sends.

A reference that drifts is worse than none: someone builds a dashboard on a
field name that has not existed for two releases and the graph is empty with
no error anywhere. So the document is checked against the code that produces
the payloads — every field the builders emit must appear in it, and every
topic the publisher can use.
"""

from __future__ import annotations

from pathlib import Path

DOC = (Path(__file__).resolve().parent.parent / "docs" / "mqtt-schema.md").read_text()


class TestEveryTopicIsDocumented:
    def test_the_metric_topics(self):
        for suffix in ("system", "gpu", "models", "fabric", "safety",
                       "transfers", "status"):
            assert f"<prefix>/<node_id>/{suffix}" in DOC, suffix

    def test_the_per_instance_topics(self):
        assert "<prefix>/<node_id>/engine/<modell>" in DOC
        assert "<prefix>/<node_id>/events/launch" in DOC

    def test_the_cluster_topic_and_why_it_has_no_node(self):
        assert "<prefix>/cluster" in DOC
        assert "Head" in DOC

    def test_the_log_topics(self):
        assert "<prefix>/<node_id>/logs/ainode" in DOC
        assert "<prefix>/<node_id>/logs/vllm/<modell>" in DOC


class TestEveryFieldIsDocumented:
    def test_the_identity_every_payload_carries(self):
        from ainode.core.config import NodeConfig
        from ainode.telemetry.payloads import node_identity

        for field in node_identity(NodeConfig(node_id="x")):
            assert f"`{field}`" in DOC, field

    def test_the_system_payload(self):
        from ainode.metrics.system import SystemSampler

        sampler = SystemSampler()
        sampler.sample()
        for group, values in sampler.sample().items():
            assert group in DOC, group
            if isinstance(values, dict) and group in ("cpu", "memory"):
                for field in values:
                    assert field in DOC, f"{group}.{field}"

    def test_the_gpu_payload(self):
        for field in ("utilization_percent", "memory_used_mb",
                      "memory_total_mb", "temperature_c"):
            assert f"`{field}`" in DOC or field in DOC, field

    def test_the_models_payload(self):
        for field in ("loaded", "embeddings", "requests_total", "errors_total",
                      "uptime_seconds", "per_model", "avg_tokens_per_second",
                      "avg_latency_ms", "tokens_generated", "load_phase",
                      "api_port", "max_model_len", "gpu_memory_utilization"):
            assert field in DOC, field

    def test_the_cluster_payload(self):
        for field in ("nodes_total", "nodes_online", "vram_total_gb",
                      "gpu_memory_used_percent", "gpu_utilization_percent",
                      "instances"):
            assert field in DOC, field

    def test_the_log_payload(self):
        from ainode.core.config import NodeConfig
        from ainode.telemetry.logs import _payload

        payload = _payload(NodeConfig(node_id="x"), "vllm", "m", ["a"], 3)
        for field in payload:
            assert f"`{field}`" in DOC, field
        assert "`skipped_bytes`" in DOC

    def test_every_load_phase_is_named(self):
        from ainode.engine.load_phase import LOAD_PHASE_ORDER

        for phase in LOAD_PHASE_ORDER:
            if phase == "idle":
                continue          # not a phase of a launch
            assert phase in DOC, phase


class TestTheNewerPayloadsAreDocumented:
    def test_the_fabric_fields(self):
        for field in ("tx_bytes", "rx_bytes", "tx_mbit_s", "tx_percent",
                      "errors_total", "errors_new", "phys_state", "link_gbit"):
            assert field in DOC, field

    def test_the_word_trap_is_spelled_out(self):
        # port_xmit_data counts 32-bit words. A figure a quarter of the truth
        # looks plausible enough to build a dashboard on.
        assert "port_xmit_data" in DOC
        assert "32-Bit" in DOC or "32-bit" in DOC

    def test_the_safety_fields(self):
        for field in ("memory_guard_enabled", "host_available_mb", "warn_mb",
                      "critical_mb", "blocking_launches", "below_critical",
                      "stops_total", "last_stop"):
            assert field in DOC, field

    def test_the_engine_fields(self):
        from ainode.telemetry.engine_metrics import WANTED

        for field, _names, _scale in WANTED:
            assert field in DOC, field
        for derived in ("requests_total_in_flight", "spec_acceptance_rate",
                        "time_to_first_token_s", "e2e_latency_s"):
            assert derived in DOC, derived

    def test_the_launch_event_fields(self):
        for field in ("outcome", "seconds", "timeline", "error"):
            assert field in DOC, field

    def test_the_transfer_fields(self):
        for field in ("downloads", "mirror", "percent", "total_bytes"):
            assert field in DOC, field

    def test_the_version_drift_fields(self):
        for field in ("versions_agree", "versions"):
            assert field in DOC, field

    def test_the_latency_percentiles(self):
        assert "latency_ms" in DOC and "p95" in DOC


class TestTheLimitsAreStated:
    def test_the_interval_bounds(self):
        from ainode.telemetry.mqtt import MAX_INTERVAL, MIN_INTERVAL

        assert str(MIN_INTERVAL) in DOC and str(MAX_INTERVAL) in DOC

    def test_the_log_caps(self):
        from ainode.telemetry.logs import DEFAULT_MAX_LINES

        assert str(DEFAULT_MAX_LINES) in DOC
        assert "64 KB" in DOC

    def test_the_topic_slug_rule(self):
        # Someone subscribing by model name needs to know what happened to
        # the slash.
        assert "A-Za-z0-9._-" in DOC


class TestTheHardwareCaveats:
    def test_it_says_the_gb10_memory_is_shared(self):
        # The single most important thing to know before alerting on either
        # memory figure.
        assert "GB10" in DOC
        assert "kein getrenntes VRAM" in DOC or "denselben Speicher" in DOC \
            or "derselbe" in DOC

    def test_it_explains_the_last_will(self):
        # The one mechanism that answers "is this node alive" without a
        # timeout in every dashboard.
        assert "Last Will" in DOC
        assert "Broker" in DOC

    def test_it_says_which_thresholds_are_enforced(self):
        # The reserve is capped against the size of the machine; an alert on
        # the configured number fires at the wrong moment.
        assert "durchgesetzten" in DOC

    def test_it_warns_about_retain(self):
        # A retained message from a node that has gone away looks alive.
        assert "Retain" in DOC or "retain" in DOC
        assert "verschwundenen Knotens" in DOC

    def test_it_says_a_missing_field_is_not_a_zero(self):
        assert "gemessen und null" in DOC
