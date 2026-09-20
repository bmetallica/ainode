"""What the engine reports, as opposed to what our proxy measured.

Everything published about a model so far was counted at the proxy: requests
through it, their latency, tokens back. That describes the traffic, not the
engine — and the questions an operator has during a busy hour are inside
vLLM. Two of them were live issues on this cluster and visible nowhere but a
log file: the KV cache filling up, and requests being preempted because of
it.
"""

from __future__ import annotations

import asyncio

from ainode.telemetry.engine_metrics import collect, distil, parse_prometheus

SAMPLE = """\
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="org/m"} 7.0
vllm:num_requests_waiting{model_name="org/m"} 3.0
vllm:gpu_cache_usage_perc{model_name="org/m"} 0.8137
vllm:num_preemptions_total{model_name="org/m"} 42.0
vllm:prompt_tokens_total{model_name="org/m"} 1000000.0
vllm:generation_tokens_total{model_name="org/m"} 412093.0
vllm:time_to_first_token_seconds_sum{model_name="org/m"} 120.0
vllm:time_to_first_token_seconds_count{model_name="org/m"} 400.0
vllm:e2e_request_latency_seconds_sum{model_name="org/m"} 840.0
vllm:e2e_request_latency_seconds_count{model_name="org/m"} 400.0
"""


class TestParsing:
    def test_comments_and_blanks_are_skipped(self):
        assert parse_prometheus("# HELP x\n\n# TYPE x gauge\n") == {}

    def test_labels_are_collapsed(self):
        # An instance serves one model, so vLLM's model_name label carries no
        # information here — and a finished request's metrics are split
        # across label sets that only make sense added up.
        metrics = parse_prometheus(
            'vllm:x{a="1"} 2.0\nvllm:x{a="2"} 3.0\n')
        assert metrics["vllm:x"] == 5.0

    def test_a_malformed_line_is_skipped_not_fatal(self):
        metrics = parse_prometheus("vllm:good 1.0\nthis is not a metric\n")
        assert metrics == {"vllm:good": 1.0}

    def test_nothing_in_is_nothing_out(self):
        assert parse_prometheus("") == {}
        assert parse_prometheus(None) == {}


class TestDistilling:
    def test_the_kv_cache_is_a_percentage(self):
        # The gauge is a fraction; every dashboard wants percent.
        assert distil(SAMPLE)["kv_cache_percent"] == 81.37

    def test_the_queue_is_reported_both_ways(self):
        out = distil(SAMPLE)
        assert out["requests_running"] == 7
        assert out["requests_waiting"] == 3
        assert out["requests_total_in_flight"] == 10

    def test_preemptions_are_carried(self):
        # The early warning before "the model suddenly got slow": requests
        # pushed out of the cache and recomputed later.
        assert distil(SAMPLE)["preemptions_total"] == 42

    def test_histograms_become_averages(self):
        # A full bucket set is dozens of lines per histogram and unusable
        # without a Prometheus behind it. sum/count is what a gauge wants.
        out = distil(SAMPLE)
        assert out["time_to_first_token_s"] == 0.3
        assert out["e2e_latency_s"] == 2.1

    def test_a_histogram_with_no_observations_is_left_out(self):
        # count 0 would be a division by zero, and an average of nothing is
        # not zero.
        out = distil("vllm:e2e_request_latency_seconds_sum 0.0\n"
                     "vllm:e2e_request_latency_seconds_count 0.0\n")
        assert "e2e_latency_s" not in out

    def test_speculative_decoding_gets_its_acceptance_rate(self):
        # The only figure that says whether the drafter is earning its keep.
        out = distil("vllm:spec_decode_num_accepted_tokens_total 300\n"
                     "vllm:spec_decode_num_draft_tokens_total 400\n")
        assert out["spec_acceptance_rate"] == 0.75

    def test_a_build_that_renamed_a_metric_still_works(self):
        # Names move between vLLM builds; each field lists the spellings seen.
        assert distil("vllm:kv_cache_usage_perc 0.5\n")["kv_cache_percent"] == 50.0

    def test_an_engine_that_reports_nothing_yields_nothing(self):
        assert distil("") == {}
        assert distil("# HELP only\n") == {}


class _Resp:
    def __init__(self, text, status=200):
        self._text = text
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text


class _Session:
    def __init__(self, by_port, fail_ports=()):
        self._by_port = by_port
        self._fail = set(fail_ports)
        self.calls = []

    def get(self, url, timeout=None):
        self.calls.append(url)
        port = int(url.split(":")[2].split("/")[0])
        if port in self._fail:
            raise OSError("connection refused")
        return _Resp(self._by_port.get(port, ""))


class _Record:
    def __init__(self, model, port):
        self.model = model
        self.api_port = port


class _Instance:
    def __init__(self, model, port):
        self.record = _Record(model, port)


class _Manager:
    def __init__(self, pairs):
        self._instances = [_Instance(m, p) for m, p in pairs]

    def instances(self):
        return list(self._instances)


class TestCollecting:
    def test_one_entry_per_instance(self):
        app = {"client_session": _Session({8000: SAMPLE, 8001: SAMPLE}),
               "instances": _Manager([("org/one", 8000), ("org/two", 8001)])}
        out = asyncio.run(collect(app))
        assert sorted(out) == ["org_one", "org_two"]
        assert out["org_one"]["requests_running"] == 7

    def test_an_engine_still_loading_costs_only_its_own_entry(self):
        # No /metrics yet is normal during a load, and a dead one has none
        # either. Both are reported elsewhere.
        app = {"client_session": _Session({8000: SAMPLE}, fail_ports=(8001,)),
               "instances": _Manager([("org/one", 8000), ("org/two", 8001)])}
        out = asyncio.run(collect(app))
        assert list(out) == ["org_one"]

    def test_no_session_is_silence(self):
        assert asyncio.run(collect({})) == {}

    def test_the_model_id_is_topic_safe(self):
        app = {"client_session": _Session({8000: SAMPLE}),
               "instances": _Manager([("org/m-1.5B", 8000)])}
        assert list(asyncio.run(collect(app))) == ["org_m-1.5B"]

    def test_it_falls_back_to_the_primary_when_there_is_no_manager(self):
        class _Config:
            model = "org/solo"
            api_port = 8000

        app = {"client_session": _Session({8000: SAMPLE}), "config": _Config()}
        assert list(asyncio.run(collect(app))) == ["org_solo"]


class TestItIsPublished:
    def test_the_loop_publishes_one_topic_per_instance(self):
        import inspect

        from ainode.telemetry.mqtt import MqttPublisher

        source = inspect.getsource(MqttPublisher._run)
        assert 'payloads[f"engine/{name}"]' in source
        assert "await collect(self._app)" in source

    def test_it_is_not_in_build_payloads(self):
        # "Publish now" and the settings preview both call build_payloads;
        # scraping every instance there would make a preview an engine load.
        import inspect

        from ainode.telemetry import payloads

        assert "engine_metrics" not in inspect.getsource(payloads.build_payloads)
