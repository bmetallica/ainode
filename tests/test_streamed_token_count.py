"""Tokens of a streamed answer are counted from vLLM's usage event.

Counting SSE events counted steps. With MTP a step yields 2.8 tokens on
Qwen3.8-Flash-Next FP4-Local, so the export said 20.1 tok/s where vLLM's own
decode time said 62.
"""

from __future__ import annotations

import json

import aiohttp
import pytest
from aiohttp import web

from ainode.api import server
from ainode.core.config import NodeConfig
from ainode.metrics.collector import MetricsCollector


def _event(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


async def _upstream(request):
    body = await request.json()
    request.app["seen"].append(body)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    # Three steps of three tokens each, split awkwardly across writes.
    data = b"".join(_event({"choices": [{"index": 0, "delta": {"content": "abc"}}]})
                    for _ in range(3))
    await resp.write(data[:17])
    await resp.write(data[17:])
    if (body.get("stream_options") or {}).get("include_usage"):
        await resp.write(_event({"choices": [], "usage": {"prompt_tokens": 5,
                                                          "completion_tokens": 9}}))
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def _proxy(aiohttp_server, aiohttp_client):
    upstream = web.Application()
    upstream["seen"] = []
    upstream.router.add_post("/v1/chat/completions", _upstream)
    up = await aiohttp_server(upstream)

    app = web.Application()
    app["config"] = NodeConfig(node_id="n", api_port=up.port)
    app["metrics_collector"] = MetricsCollector()

    async def _session(app):
        app["client_session"] = aiohttp.ClientSession()
        yield
        await app["client_session"].close()

    app.cleanup_ctx.append(_session)
    app.router.add_post("/v1/chat/completions", server.proxy_to_vllm)
    client = await aiohttp_client(app)
    return client, app, upstream


def _tokens(app):
    return app["metrics_collector"].model_stats()["M"]["tokens_generated"]


@pytest.mark.asyncio
async def test_the_count_is_vllms_and_the_client_sees_no_extra_event(aiohttp_server, aiohttp_client):
    client, app, upstream = await _proxy(aiohttp_server, aiohttp_client)
    resp = await client.post("/v1/chat/completions",
                             json={"model": "M", "stream": True, "messages": []})
    text = await resp.text()
    assert upstream["seen"][0]["stream_options"] == {"include_usage": True}
    assert '"usage"' not in text                       # held back: not asked for
    assert text.count('"abc"') == 3 and text.endswith("data: [DONE]\n\n")
    assert _tokens(app) == 9                            # not 4 events


@pytest.mark.asyncio
async def test_a_client_that_asked_gets_its_usage_event(aiohttp_server, aiohttp_client):
    client, app, upstream = await _proxy(aiohttp_server, aiohttp_client)
    resp = await client.post("/v1/chat/completions",
                             json={"model": "M", "stream": True, "messages": [],
                                   "stream_options": {"include_usage": True}})
    text = await resp.text()
    assert '"completion_tokens": 9' in text
    assert _tokens(app) == 9


def test_only_streaming_completions_are_touched():
    body = json.dumps({"model": "M", "messages": []}).encode()
    assert server._ask_for_usage("/v1/chat/completions", body) == (body, False)
    streaming = json.dumps({"model": "M", "stream": True}).encode()
    assert server._ask_for_usage("/v1/embeddings", streaming) == (streaming, False)
    assert server._ask_for_usage("/v1/completions", streaming)[1] is True
