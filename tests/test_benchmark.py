"""Measuring throughput from the UI, instead of from a throwaway script.

Two numbers from one afternoon on this cluster made the case:

    Gemma 4 26B, one node      70.3 tok/s at 1 stream, 544.0 at 16
    MiniMax M2.7, two nodes    20.5 tok/s at 1 stream — its catalog entry
                               claimed ~42, an unverified community figure

Neither is derivable from a parameter count, and the second contradicted what
the catalog said. So the measurement belongs in the product: the cluster
answers "how many users" better than any estimate does.

Context size is selectable because it changes the answer. Throughput at a 1K
prompt and at a 64K prompt are different numbers, and the second is what RAG
and coding workloads actually look like.
"""

from __future__ import annotations

import asyncio
import json
from unittest import mock

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from ainode.api.server import create_app
from ainode.bench.runner import (
    MAX_CONCURRENCY,
    BenchError,
    BenchSpec,
    build_prompt,
    run_benchmark,
)
from ainode.core.config import NodeConfig


class TestTheSpecIsChecked:
    def test_levels_are_sorted_and_deduplicated(self):
        spec = BenchSpec(model="m", concurrency=[8, 1, 4, 4]).validate()
        assert spec.concurrency == [1, 4, 8]

    def test_no_model_is_refused(self):
        with pytest.raises(BenchError, match="no model"):
            BenchSpec(model="").validate()

    def test_a_runaway_concurrency_is_refused(self):
        """This points a load generator at a production cluster; a typo in the
        UI should not hold every KV slot for an hour."""
        with pytest.raises(BenchError, match="refused"):
            BenchSpec(model="m", concurrency=[MAX_CONCURRENCY + 1]).validate()

    def test_too_many_levels_are_refused(self):
        with pytest.raises(BenchError, match="levels"):
            BenchSpec(model="m", concurrency=list(range(1, 12))).validate()

    def test_absurd_output_lengths_are_refused(self):
        with pytest.raises(BenchError, match="max_tokens"):
            BenchSpec(model="m", max_tokens=0).validate()
        with pytest.raises(BenchError, match="max_tokens"):
            BenchSpec(model="m", max_tokens=99999).validate()

    def test_an_unknown_prompt_style_is_refused(self):
        with pytest.raises(BenchError, match="style"):
            BenchSpec(model="m", style="poetry").validate()


class TestThePrompt:
    def test_a_short_prompt_is_just_the_question(self):
        assert len(build_prompt("chat", 0)) < 400

    @pytest.mark.parametrize("tokens", [4096, 32768])
    def test_padding_lands_near_the_requested_size(self, tokens):
        # Four characters per token is an estimate; the RESULT reports what
        # the engine actually counted, so this only has to be close.
        estimated = len(build_prompt("rag", tokens)) / 4
        assert 0.85 * tokens <= estimated <= 1.3 * tokens

    def test_the_padding_is_not_one_repeated_sentence(self):
        """A run of identical tokens is unrepresentative, and prefix caching
        would collapse it — reporting a context length never attended over."""
        prompt = build_prompt("chat", 8192)
        chunks = [prompt[i:i + 120] for i in range(0, len(prompt) - 600, 600)]
        assert len(set(chunks)) == len(chunks)

    def test_the_question_survives_the_padding(self):
        assert "CSV" in build_prompt("code", 16384)


def _reply(tokens: int = 100, prompt_tokens: int = 20, status: int = 200,
           error: str = ""):
    payload = ({"error": {"message": error}} if error else
               {"choices": [{"message": {"content": "x"}}],
                "usage": {"completion_tokens": tokens,
                          "prompt_tokens": prompt_tokens}})

    class _Response:
        def __init__(self):
            self.status = status

        async def json(self):
            return payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, *a, **k):
            return _Response()

    return _Session()


class TestRunning:
    @pytest.mark.asyncio
    async def test_each_level_is_measured(self):
        spec = BenchSpec(model="m", concurrency=[1, 4])
        results = await run_benchmark(spec, session=_reply())
        assert [r.concurrency for r in results] == [1, 4]
        assert results[1].ok == 4
        assert results[1].completion_tokens == 400

    @pytest.mark.asyncio
    async def test_the_measured_prompt_size_is_reported(self):
        """Not the requested one. Four characters per token is an estimate,
        and an estimate presented as a context length is how a capacity plan
        goes wrong."""
        spec = BenchSpec(model="m", concurrency=[1], prompt_tokens=4096)
        results = await run_benchmark(spec, session=_reply(prompt_tokens=4211))
        assert results[0].prompt_tokens == 4211

    @pytest.mark.asyncio
    async def test_levels_do_not_overlap(self):
        """Two levels in flight at once measure each other, not the model."""
        active = {"now": 0, "peak": 0}

        class _Session:
            def post(self, *a, **k):

                class _R:
                    status = 200

                    async def json(self):
                        return {"usage": {"completion_tokens": 10,
                                          "prompt_tokens": 5}}

                    async def __aenter__(self):
                        active["now"] += 1
                        active["peak"] = max(active["peak"], active["now"])
                        await asyncio.sleep(0)
                        return self

                    async def __aexit__(self, *a):
                        active["now"] -= 1
                        return False

                return _R()

        await run_benchmark(BenchSpec(model="m", concurrency=[1, 4]),
                            session=_Session())
        assert active["peak"] == 4          # not 5

    @pytest.mark.asyncio
    async def test_a_failing_level_reports_the_reason(self):
        results = await run_benchmark(
            BenchSpec(model="m", concurrency=[1]),
            session=_reply(status=400, error="context length exceeded"))
        assert results[0].failed == 1
        assert "context length exceeded" in results[0].error

    @pytest.mark.asyncio
    async def test_a_total_failure_stops_early(self):
        """Every request failing is a configuration problem, not a data point,
        and the heavier levels would take minutes to repeat it."""
        results = await run_benchmark(
            BenchSpec(model="m", concurrency=[1, 4, 8, 16]),
            session=_reply(status=404, error="model not found"))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_progress_is_reported_per_level(self):
        seen = []
        await run_benchmark(BenchSpec(model="m", concurrency=[1, 4]),
                            session=_reply(), on_progress=seen.append)
        assert [e["state"] for e in seen] == [
            "running", "level_done", "running", "level_done"]


@pytest_asyncio.fixture
async def client():
    app = create_app(config=NodeConfig(node_id="n1", model="org/model"),
                     engine=None)
    async with TestClient(TestServer(app)) as c:
        yield c


class TestTheEndpoints:
    @pytest.mark.asyncio
    async def test_options_lists_what_can_be_measured(self, client):
        payload = await (await client.get("/api/bench/options")).json()
        assert "org/model" in payload["models"]
        assert set(payload["styles"]) == {"chat", "code", "rag"}
        assert payload["limits"]["concurrency"] == MAX_CONCURRENCY

    @pytest.mark.asyncio
    async def test_a_bad_spec_is_a_400_not_a_failed_run(self, client):
        response = await client.post("/api/bench/run",
                                     json={"model": "m", "max_tokens": 0})
        assert response.status == 400
        assert "max_tokens" in (await response.json())["error"]

    @pytest.mark.asyncio
    async def test_a_run_is_a_job_not_a_held_connection(self, client):
        with mock.patch("ainode.bench.api_routes.run_benchmark",
                        new=mock.AsyncMock(return_value=[])):
            response = await client.post("/api/bench/run",
                                         json={"model": "m", "concurrency": [1]})
            assert response.status == 202
            payload = await response.json()
        assert payload["started"] is True
        assert payload["spec"]["concurrency"] == [1]

    @pytest.mark.asyncio
    async def test_two_at_once_are_refused(self, client):
        started = asyncio.Event()

        async def _slow(*a, **k):
            started.set()
            await asyncio.sleep(5)
            return []

        with mock.patch("ainode.bench.api_routes.run_benchmark", new=_slow):
            assert (await client.post("/api/bench/run",
                                      json={"model": "m"})).status == 202
            await asyncio.wait_for(started.wait(), timeout=2)
            second = await client.post("/api/bench/run", json={"model": "m"})
            assert second.status == 409
            assert "already running" in (await second.json())["error"]
            await client.post("/api/bench/cancel")

    @pytest.mark.asyncio
    async def test_the_409_is_json(self, client):
        """The state carries an asyncio.Task. Passing it straight out turned
        "another run is in progress" — an ordinary answer — into a 500."""
        started = asyncio.Event()

        async def _slow(*a, **k):
            started.set()
            await asyncio.sleep(5)
            return []

        with mock.patch("ainode.bench.api_routes.run_benchmark", new=_slow):
            await client.post("/api/bench/run", json={"model": "m"})
            await asyncio.wait_for(started.wait(), timeout=2)
            second = await client.post("/api/bench/run", json={"model": "m"})
            assert second.status == 409
            assert "task" not in (await second.json())["status"]
            await client.post("/api/bench/cancel")

    @pytest.mark.asyncio
    async def test_status_is_readable_before_anything_ran(self, client):
        payload = await (await client.get("/api/bench/status")).json()
        assert payload["running"] is False

    @pytest.mark.asyncio
    async def test_the_task_handle_does_not_cross_the_wire(self, client):
        """asyncio.Task is not JSON, and a 500 from the status poll would look
        like the benchmark itself had failed."""
        with mock.patch("ainode.bench.api_routes.run_benchmark",
                        new=mock.AsyncMock(return_value=[])):
            await client.post("/api/bench/run", json={"model": "m"})
        body = await (await client.get("/api/bench/status")).text()
        assert "task" not in json.loads(body)

    @pytest.mark.asyncio
    async def test_cancelling_nothing_is_not_an_error(self, client):
        payload = await (await client.post("/api/bench/cancel")).json()
        assert payload["cancelled"] is False

    @pytest.mark.asyncio
    async def test_a_running_benchmark_can_be_stopped(self, client):
        """It holds KV slots a real user wants back, and the levels above the
        current one can take minutes each."""
        started = asyncio.Event()

        async def _slow(*a, **k):
            started.set()
            await asyncio.sleep(30)
            return []

        with mock.patch("ainode.bench.api_routes.run_benchmark", new=_slow):
            await client.post("/api/bench/run", json={"model": "m"})
            await asyncio.wait_for(started.wait(), timeout=2)
            assert (await (await client.post("/api/bench/cancel")).json())["cancelled"]
            for _ in range(20):
                await asyncio.sleep(0.05)
                if not (await (await client.get("/api/bench/status")).json())["running"]:
                    break
            payload = await (await client.get("/api/bench/status")).json()
        assert payload["running"] is False
        assert payload["error"] == "cancelled"


class TestTheUi:
    def _source(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                "static" / "js" / "app.js").read_text()

    def test_context_sizes_are_selectable(self):
        # The point of the feature: throughput at 1K and at 64K are different
        # numbers, and the second is the realistic one.
        source = self._source()
        assert "BENCH_CONTEXTS" in source
        assert "'64K context', tokens: 65536" in source

    def test_the_panel_warns_what_it_costs(self):
        assert "real requests will " in self._source()

    def test_it_reads_the_curve_not_just_the_numbers(self):
        """"Still scaling" and "saturated" call for different remedies, which
        is the whole reason to run this."""
        source = self._source()
        assert "_benchReading" in source
        assert "Still scaling" in source
        assert "Saturated" in source
