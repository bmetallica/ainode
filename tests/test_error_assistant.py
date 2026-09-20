"""The error assistant: a diagnosis from a model that is already running.

The rules this file pins down:

* the raw error is never replaced — the assistant's output is additive;
* no model loaded means no assistant, and no button;
* the failing model is never the one asked;
* the prompt carries what a public model cannot know: what AINode is, what
  this hardware is, how the instance was launched, and the part of the engine
  log that is about the failure;
* nothing here can turn a failure into a second failure — an unreachable
  peer, an unreadable log or a helper that errors all degrade to less context
  or a plain message.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ainode.assist.briefing import SYSTEM_BRIEFING, hardware_notes
from ainode.assist.context import collect_context, log_excerpt, render_context
from ainode.assist.diagnose import (
    AssistError,
    MAX_CONTEXT_CHARS,
    ask_helper,
    build_messages,
    choose_helper,
    helper_candidates,
)


class _Node:
    def __init__(self, node_id, model="", instances=None, gpu="NVIDIA GB10",
                 status="online", fabric_ip="10.0.0.2", web_port=3000,
                 api_port=8000):
        self.node_id = node_id
        self.node_name = node_id.upper()
        self.model = model
        self.instances = instances or []
        self.gpu_name = gpu
        self.gpu_memory_gb = 128.0
        self.gpu_memory_used_mb = 64000.0
        self.gpu_memory_total_mb = 128000.0
        self.status = status
        self.fabric_ip = fabric_ip
        self.web_port = web_port
        self.api_port = api_port
        self.embedding_models = []


class _Cluster:
    def __init__(self, nodes):
        self._nodes = nodes

    def members(self):
        return list(self._nodes)

    def get_node(self, node_id):
        return next((n for n in self._nodes if n.node_id == node_id), None)


class _Config:
    node_id = "head"
    api_port = 8000


def _app(nodes):
    return {"config": _Config(), "cluster_state": _Cluster(nodes)}


READY = {"model": "org/chat", "api_port": 8000, "status": "serving",
         "load_phase": "ready"}


class TestPickingAHelper:
    def test_nothing_loaded_means_no_assistant(self):
        assert helper_candidates(_app([_Node("head")])) == []

    def test_the_failing_model_is_never_asked(self):
        app = _app([_Node("head", instances=[READY])])
        assert helper_candidates(app, exclude="org/chat") == []

    def test_a_local_model_is_preferred_over_a_remote_one(self):
        app = _app([
            _Node("head", instances=[dict(READY, model="org/local")]),
            _Node("n2", instances=[dict(READY, model="org/remote")]),
        ])
        assert [c[0] for c in helper_candidates(app)] == ["org/local", "org/remote"]
        assert helper_candidates(app)[0][1] == "localhost"

    def test_the_choice_is_stable(self):
        # Two clicks a second apart must not consult two different models.
        app = _app([_Node("head", instances=[
            dict(READY, model="org/b", api_port=8001),
            dict(READY, model="org/a")])])
        assert choose_helper(app)[0] == choose_helper(app)[0] == "org/a"

    def test_a_named_helper_is_honoured(self):
        app = _app([_Node("head", instances=[
            dict(READY, model="org/a"), dict(READY, model="org/b", api_port=8001)])])
        assert choose_helper(app, wanted="org/b")[0] == "org/b"

    def test_a_named_helper_that_is_not_serving_is_not_substituted(self):
        app = _app([_Node("head", instances=[dict(READY, model="org/a")])])
        assert choose_helper(app, wanted="org/gone") is None

    def test_a_broken_cluster_state_is_not_a_crash(self):
        class _Boom:
            def members(self):
                raise RuntimeError("no")
        assert helper_candidates({"config": _Config(), "cluster_state": _Boom()}) == []


class TestTheLogExcerpt:
    def test_progress_bars_are_dropped(self):
        text = "\n".join(["Loading safetensors checkpoint shards:  40%|#### | 2/5"] * 200
                         + ["ValueError: boom"])
        out = log_excerpt(text)
        assert "%|" not in out
        assert "ValueError: boom" in out

    def test_a_traceback_far_above_the_tail_survives(self):
        # The whole point: the failure is 300 lines up, behind the noise.
        text = "\n".join(
            ["Traceback (most recent call last):", "ValueError: unsupported"]
            + [f"INFO idle line {i}" for i in range(400)])
        out = log_excerpt(text, max_lines=20)
        assert "ValueError: unsupported" in out
        assert "(log truncated)" in out

    def test_the_tail_is_kept(self):
        text = "\n".join(f"line {i}" for i in range(100))
        assert "line 99" in log_excerpt(text, max_lines=10)
        assert "line 0" not in log_excerpt(text, max_lines=10)

    def test_it_is_bounded(self):
        text = "\n".join("x" * 200 for _ in range(500))
        assert len(log_excerpt(text, max_chars=1000)) <= 1100

    def test_nothing_in_is_nothing_out(self):
        assert log_excerpt("") == ""
        assert log_excerpt(None) == ""


class TestTheContext:
    def _context(self, **kw):
        app = _app([_Node("head", instances=[READY]), _Node("n2")])
        return collect_context(
            app,
            kw.get("model", "org/big"),
            kw.get("node_id", "head"),
            kw.get("error", "CUDA out of memory"),
            launch=kw.get("launch"),
            log_text=kw.get("log", "Traceback (most recent call last):\nboom"),
        )

    def test_the_operators_error_is_carried_verbatim(self):
        rendered = render_context(self._context(error="Unsupported weight_bits: 16"))
        assert "Unsupported weight_bits: 16" in rendered

    def test_the_nodes_and_their_memory_are_described(self):
        rendered = render_context(self._context())
        assert "128 GB" in rendered and "50% in use" in rendered

    def test_the_launch_settings_are_named(self):
        rendered = render_context(self._context(launch={
            "node_ids": ["head", "n2"], "strategy": "tensor",
            "max_model_len": 65536, "kv_cache_dtype": "fp8",
            "extra_vllm_args": ["--enforce-eager"]}))
        assert "max_model_len: 65536" in rendered
        assert "--enforce-eager" in rendered

    def test_an_unset_setting_is_not_invented(self):
        rendered = render_context(self._context(launch={"model": "org/big"}))
        assert "defaults" in rendered
        assert "None" not in rendered

    def test_a_missing_launch_config_says_so(self):
        # A launch that died early leaves no instance. Silence here would read
        # as "the defaults applied", which is a different diagnosis.
        assert "could not be read" in render_context(self._context(launch=None))

    def test_an_unreadable_log_says_so_rather_than_pretending(self):
        assert "no log could be read" in render_context(self._context(log=""))


class TestThePrompt:
    def test_it_explains_what_ainode_is(self):
        messages = build_messages(collect_context(
            _app([_Node("head")]), "org/m", "head", "boom"))
        system = messages[0]["content"]
        assert "AINode" in system and "vLLM" in system

    def test_it_carries_the_rules_that_decide_most_failures(self):
        assert "never 3" in SYSTEM_BRIEFING          # TP is a power of two
        assert "compiled-kernel cache" in SYSTEM_BRIEFING
        assert "gpu-memory-utilization" in SYSTEM_BRIEFING

    def test_gb10_facts_only_appear_on_gb10(self):
        assert "unified" in hardware_notes(["NVIDIA GB10"])
        assert hardware_notes(["NVIDIA RTX 6000 Ada"]) == ""
        assert hardware_notes([]) == ""

    def test_the_gb10_note_warns_that_missing_memory_readings_are_normal(self):
        # Otherwise every diagnosis on this hardware blames nvidia-smi's [N/A].
        assert "[N/A]" in hardware_notes(["NVIDIA GB10"])

    def test_a_huge_log_cannot_overflow_a_small_helpers_window(self):
        context = collect_context(_app([_Node("head")]), "org/m", "head", "boom",
                                  log_text="x" * 200000)
        user = build_messages(context)[1]["content"]
        assert len(user) < MAX_CONTEXT_CHARS + 200

    def test_the_model_is_told_not_to_repeat_the_error_back(self):
        assert "do not repeat it back" in SYSTEM_BRIEFING


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else {}
        self._text = text or json.dumps(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text

    async def json(self, content_type=None):
        return self._payload


class _Session:
    def __init__(self, resp=None, raises=None):
        self._resp = resp
        self._raises = raises
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        if self._raises:
            raise self._raises
        return self._resp


def _answer(content, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message}], "usage": {"prompt_tokens": 10}}


class TestAskingTheHelper:
    def test_a_plain_answer(self):
        session = _Session(_Resp(payload=_answer("Lower --max-model-len.")))
        out = asyncio.run(ask_helper(session, "org/chat", "localhost", 8000, []))
        assert out["answer"] == "Lower --max-model-len."
        assert out["helper_model"] == "org/chat"
        assert session.calls[0][0] == "http://localhost:8000/v1/chat/completions"

    def test_it_does_not_stream(self):
        # The card renders one block; a stream would have nowhere to go.
        session = _Session(_Resp(payload=_answer("ok")))
        asyncio.run(ask_helper(session, "m", "localhost", 8000, []))
        assert session.calls[0][1]["stream"] is False

    def test_a_reasoning_model_that_spent_its_budget_thinking_still_answers(self):
        # Empty content with a full reasoning block is the exact shape that
        # makes a chat client believe the turn ended. Show the thinking rather
        # than reporting nothing.
        session = _Session(_Resp(payload=_answer("", reasoning="It ran out of KV.")))
        out = asyncio.run(ask_helper(session, "m", "localhost", 8000, []))
        assert out["answer"] == "It ran out of KV."

    def test_content_parts_are_joined(self):
        payload = {"choices": [{"message": {"content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
        out = asyncio.run(ask_helper(_Session(_Resp(payload=payload)),
                                     "m", "localhost", 8000, []))
        assert out["answer"] == "ab"

    def test_an_http_error_names_the_model(self):
        session = _Session(_Resp(status=400, text="bad request"))
        with pytest.raises(AssistError) as exc:
            asyncio.run(ask_helper(session, "org/chat", "localhost", 8000, []))
        assert "org/chat" in str(exc.value)

    def test_an_unreachable_helper_is_reported_not_raised_as_noise(self):
        session = _Session(raises=OSError("connection refused"))
        with pytest.raises(AssistError) as exc:
            asyncio.run(ask_helper(session, "m", "10.0.0.9", 8000, []))
        assert "10.0.0.9" in str(exc.value)

    def test_an_empty_answer_is_an_error_not_a_blank_card(self):
        with pytest.raises(AssistError):
            asyncio.run(ask_helper(_Session(_Resp(payload=_answer(""))),
                                   "m", "localhost", 8000, []))


class _Req:
    def __init__(self, app, body=None, query=None):
        self.app = app
        self._body = body
        self.query = query or {}

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class TestTheRoutes:
    def test_status_says_no_when_nothing_is_loaded(self):
        from ainode.assist.api_routes import handle_assist_status

        resp = asyncio.run(handle_assist_status(_Req(_app([_Node("head")]))))
        assert json.loads(resp.body)["available"] is False

    def test_status_excludes_the_failing_model(self):
        from ainode.assist.api_routes import handle_assist_status

        app = _app([_Node("head", instances=[READY])])
        body = json.loads(asyncio.run(
            handle_assist_status(_Req(app, query={"exclude": "org/chat"}))).body)
        assert body["available"] is False

    def test_diagnose_without_a_helper_is_a_409_with_a_plain_message(self):
        from ainode.assist.api_routes import handle_assist_diagnose

        resp = asyncio.run(handle_assist_diagnose(
            _Req(_app([_Node("head")]), {"model": "org/big", "error": "boom"})))
        assert resp.status == 409
        assert "no other model is loaded" in json.loads(resp.body)["error"]

    def test_diagnose_needs_something_to_diagnose(self):
        from ainode.assist.api_routes import handle_assist_diagnose

        resp = asyncio.run(handle_assist_diagnose(_Req(_app([_Node("head")]), {})))
        assert resp.status == 400

    def test_the_engine_log_route_answers_with_a_tail(self, monkeypatch):
        from ainode.assist import api_routes

        class _Backend:
            def logs(self, n=100):
                return "line a\nline b"

        app = {"config": _Config(), "engine": _Backend()}
        body = json.loads(asyncio.run(
            api_routes.handle_engine_log(_Req(app, query={"lines": "50"}))).body)
        assert body["text"] == "line a\nline b"

    def test_an_unreadable_log_is_empty_not_a_500(self):
        from ainode.assist import api_routes

        class _Backend:
            def logs(self, n=100):
                raise OSError("gone")

        app = {"config": _Config(), "engine": _Backend()}
        body = json.loads(asyncio.run(api_routes.handle_engine_log(_Req(app))).body)
        assert body["text"] == ""

    def test_the_whole_path(self):
        # Error in, diagnosis out: the helper is asked, and what it was told
        # contains the operator's error and the engine log — the two things
        # that make the answer better than a generic one.
        from ainode.assist import api_routes

        class _Backend:
            def logs(self, n=100):
                return "ValueError: Unsupported weight_bits: 16"

        session = _Session(_Resp(payload=_answer("The checkpoint is mixed-bit.")))
        app = _app([_Node("head", instances=[READY])])
        app["client_session"] = session
        app["engine"] = _Backend()

        resp = asyncio.run(api_routes.handle_assist_diagnose(_Req(app, {
            "model": "org/big", "node_id": "head",
            "error": "Unsupported weight_bits: 16"})))
        body = json.loads(resp.body)
        assert body["ok"] is True
        assert body["answer"] == "The checkpoint is mixed-bit."
        assert body["helper_model"] == "org/chat"

        sent = session.calls[0][1]["messages"]
        assert "AINode" in sent[0]["content"]
        assert "Unsupported weight_bits: 16" in sent[1]["content"]
        assert "ENGINE LOG" in sent[1]["content"]

    def test_a_helper_that_fails_reports_itself_and_nothing_else_breaks(self):
        from ainode.assist import api_routes

        app = _app([_Node("head", instances=[READY])])
        app["client_session"] = _Session(raises=OSError("refused"))
        resp = asyncio.run(api_routes.handle_assist_diagnose(_Req(app, {
            "model": "org/big", "error": "boom"})))
        assert resp.status == 502
        assert json.loads(resp.body)["helper_model"] == "org/chat"

    def test_the_routes_are_registered(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert {"/api/assist/status", "/api/assist/diagnose",
                "/api/engine/log"} <= paths


WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
APP_JS = (WEB / "static" / "js" / "app.js").read_text()


class TestTheCard:
    def test_the_raw_error_is_still_rendered(self):
        # The assistant is additive. If this ever stops being true, the
        # operator loses the only authoritative text about the failure. It
        # moved to the details dialog with the rest of the detail, and it is
        # still the first thing in that section.
        assert "'<div class=\"instance-failed-note\">' + this.esc(error)" in APP_JS

    def test_no_button_without_a_loaded_model(self):
        assert "if (!helpers.length) return '';" in APP_JS

    def test_the_answer_says_which_model_produced_it(self):
        assert "Suggested by" in APP_JS

    def test_the_answer_is_escaped(self):
        assert "self.esc(entry.answer)" in APP_JS or "this.esc(entry.answer)" in APP_JS
