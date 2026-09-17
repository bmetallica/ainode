"""A client config the cluster writes about itself.

Writing one by hand needs four facts per model, three of which are easy to
get wrong in ways that surface hours later — and all three were got wrong on
this cluster before this existed:

  * whether it reasons. Told otherwise, the client ignores the reasoning
    field, sees an empty `content`, and treats the turn as finished. The model
    "stops by itself". Diagnosed on GLM, then again on DeepSeek V4 before the
    pattern was recognised.
  * whether it takes images. Offering them to a text-only model fails in the
    engine rather than in the client.
  * the context limit, which has to fit under the max_model_len the instance
    was LAUNCHED with — often far below what the model advertises, since
    DeepSeek V4 claims 1M and serves 131072 here — and has to leave room,
    because a client's token accounting is an estimate. 96000 + 32768 against
    131072 leaves 2304 tokens of it.
"""

from __future__ import annotations

import pytest

from ainode.clients.opencode import build_opencode_config, limits_for


class TestTheLimitsFitTheWindow:
    @pytest.mark.parametrize("window", [4096, 32768, 131072, 196608, 262144])
    def test_context_plus_output_stays_under_it(self, window):
        limits = limits_for(window)
        assert limits["context"] + limits["output"] < window

    @pytest.mark.parametrize("window", [32768, 131072, 262144])
    def test_the_margin_is_not_token_thin(self, window):
        """2304 tokens of headroom is what produced mid-session refusals."""
        limits = limits_for(window)
        assert window - limits["context"] - limits["output"] >= 4096

    def test_the_measured_case_lands_where_it_was_set_by_hand(self):
        """131072 was worked out on the cluster as ~88000 + 24576."""
        limits = limits_for(131072)
        assert 85000 <= limits["context"] <= 95000
        assert limits["output"] >= 16384

    def test_a_reasoning_model_gets_a_generous_output_budget(self):
        """979 reasoning tokens measured for 'count from 1 to 30'. A small
        output cap means empty answers, not short ones."""
        assert limits_for(131072)["output"] >= 16384

    def test_a_tiny_window_still_leaves_something_to_read(self):
        limits = limits_for(4096)
        assert limits["output"] >= 256 and limits["context"] >= 512


class _Entry:
    kind = "llm"

    def __init__(self, model, max_model_len=None, extra_vllm_args=()):
        self.model = model
        self.max_model_len = max_model_len
        self.extra_vllm_args = list(extra_vllm_args)


class _Info:
    def __init__(self, name="", capabilities=(), extra_vllm_args=(),
                 context_length=0):
        self.name = name
        self.capabilities = list(capabilities)
        self.extra_vllm_args = list(extra_vllm_args)
        self.context_length = context_length


class _Manager:
    def __init__(self, by_repo):
        self._by_repo = by_repo

    def _find_catalog_by_hf_repo(self, repo):
        return self._by_repo.get(repo)


def _build(entries, catalog=None, base="http://192.168.1.2:3000"):
    from unittest import mock

    app = {"model_manager": _Manager(catalog or {})}

    class _Profile:
        pass

    profile = _Profile()
    profile.entries = entries
    with mock.patch("ainode.profiles.apply.capture_profile", return_value=profile):
        return build_opencode_config(app, base)


DEEPSEEK = "deepseek-ai/DeepSeek-V4-Flash-DSpark"


class TestTheModelsSection:
    def test_capabilities_come_from_the_catalog(self):
        result = _build(
            [_Entry(DEEPSEEK, max_model_len=131072)],
            {DEEPSEEK: _Info("DeepSeek V4 Flash",
                             ["tool_use", "reasoning", "code"])})
        model = result["config"]["provider"]["vllm"]["models"][DEEPSEEK]
        assert model["reasoning"] is True
        assert model["tool_call"] is True
        assert model["attachment"] is False

    def test_a_vision_model_gets_attachments(self):
        result = _build([_Entry("org/eyes", max_model_len=32768)],
                        {"org/eyes": _Info("Eyes", ["vision", "tool_use"])})
        assert result["config"]["provider"]["vllm"]["models"]["org/eyes"][
            "attachment"] is True

    def test_the_window_comes_from_the_launch_not_the_model(self):
        """DeepSeek V4 advertises 1M and serves 131072. Taking the advertised
        figure produces a config that is refused mid-session."""
        result = _build(
            [_Entry(DEEPSEEK, extra_vllm_args=["--max-model-len", "131072"])],
            {DEEPSEEK: _Info("DeepSeek", ["reasoning"], context_length=1048576)})
        limits = result["config"]["provider"]["vllm"]["models"][DEEPSEEK]["limit"]
        assert limits["context"] + limits["output"] < 131072

    def test_the_recipes_flag_counts_too(self):
        """A catalog model carries --max-model-len in its recipe rather than
        in a per-launch override."""
        result = _build(
            [_Entry(DEEPSEEK)],
            {DEEPSEEK: _Info("DeepSeek", [],
                             extra_vllm_args=["--max-model-len", "65536"],
                             context_length=1048576)})
        limits = result["config"]["provider"]["vllm"]["models"][DEEPSEEK]["limit"]
        assert limits["context"] + limits["output"] < 65536

    def test_the_equals_form_is_read(self):
        result = _build([_Entry("a/b", extra_vllm_args=["--max-model-len=8192"])])
        limits = result["config"]["provider"]["vllm"]["models"]["a/b"]["limit"]
        assert limits["context"] + limits["output"] < 8192

    def test_no_flag_anywhere_says_so_instead_of_guessing(self):
        result = _build([_Entry("a/b")],
                        {"a/b": _Info("B", [], context_length=262144)})
        assert any("no --max-model-len" in n for n in result["notes"])

    def test_a_model_outside_the_catalog_still_appears(self):
        """Conservatively: no capabilities claimed, which fails safe — a
        missing tool_call costs a feature, a wrong one costs a session."""
        result = _build([_Entry("who/knows", max_model_len=32768)])
        model = result["config"]["provider"]["vllm"]["models"]["who/knows"]
        assert model["tool_call"] is False and model["reasoning"] is False
        assert model["name"] == "knows"


class TestTheProviderBlock:
    def test_the_base_url_is_the_openai_path(self):
        options = _build([_Entry("a/b", max_model_len=8192)])[
            "config"]["provider"]["vllm"]["options"]
        assert options["baseURL"] == "http://192.168.1.2:3000/v1"

    def test_a_trailing_slash_does_not_double_up(self):
        options = _build([_Entry("a/b", max_model_len=8192)],
                         base="http://x:3000/")["config"]["provider"]["vllm"]["options"]
        assert options["baseURL"] == "http://x:3000/v1"

    def test_an_api_key_is_present(self):
        """Some openai-compatible clients send no Authorization header
        without one and fail on the first request."""
        options = _build([_Entry("a/b", max_model_len=8192)])[
            "config"]["provider"]["vllm"]["options"]
        assert options["apiKey"]

    def test_the_roomiest_model_is_the_default(self):
        result = _build([_Entry("small/one", max_model_len=8192),
                         _Entry("big/one", max_model_len=131072)])
        assert result["config"]["model"] == "vllm/big/one"
        assert result["config"]["small_model"] == "vllm/small/one"

    def test_one_model_gets_no_small_model(self):
        result = _build([_Entry("only/one", max_model_len=8192)])
        assert "small_model" not in result["config"]

    def test_nothing_running_is_not_an_error(self):
        result = _build([])
        assert result["config"]["provider"]["vllm"]["models"] == {}
        assert "model" not in result["config"]


class TestItIsReachable:
    def test_the_route_exists(self):
        from ainode.api.server import create_app
        from ainode.core.config import NodeConfig

        app = create_app(config=NodeConfig(node_id="head"), engine=None)
        paths = {getattr(r.resource, "canonical", "") for r in app.router.routes()}
        assert "/api/clients/opencode" in paths

    def test_the_button_is_in_the_server_view(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "ainode" / "web" /
                  "static" / "js" / "app.js").read_text()
        assert 'id="opencode-config"' in source
        assert "/api/clients/opencode?base_url=" in source
