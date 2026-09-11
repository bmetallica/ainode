"""What an operator can configure without reaching for curl.

Three models served to three groups of people is the target deployment, and
every knob that shapes it — concurrency, context, KV precision, API aliases —
has to be reachable from the launch panel. The Security section matters for the
same reason: auth is off by default, so a shared cluster is open until someone
turns it on, and there was no UI to do that with.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
INDEX = (WEB / "templates" / "index.html").read_text()
APP_JS = (WEB / "static" / "js" / "app.js").read_text()
STYLE = (WEB / "static" / "css" / "style.css").read_text()


class TestLaunchPanelFields:
    @pytest.mark.parametrize(
        "element_id",
        [
            "launch-model", "launch-gmu",
            "launch-max-seqs",        # --max-num-seqs: concurrent requests
            "launch-max-len",         # max_model_len
            "launch-kv-dtype",        # fp8 vs auto — vision models need auto
            "launch-served-name",     # API alias, what OpenWebUI addresses
            "launch-quantization",
            "launch-engine-image",    # Qwen3.8 needs vllm/vllm-openai:v0.27.1
            "launch-extra-args",
            "launch-trust-remote-code",
        ],
    )
    def test_field_exists(self, element_id):
        assert f'id="{element_id}"' in INDEX

    def test_every_field_is_read_by_the_launcher(self):
        for element_id in ("launch-max-seqs", "launch-max-len", "launch-kv-dtype",
                           "launch-served-name", "launch-quantization",
                           "launch-engine-image", "launch-extra-args",
                           "launch-trust-remote-code"):
            assert f"'{element_id}'" in APP_JS, f"{element_id} is rendered but never read"

    def test_sent_field_names_match_the_api(self):
        """A field the API does not know is silently ignored, which looks like
        the setting had no effect."""
        from ainode.models.api_routes import _OVERRIDE_KEYS

        for key in ("kv_cache_dtype", "quantization", "engine_image",
                    "served_model_name", "trust_remote_code", "extra_vllm_args",
                    "max_model_len"):
            assert f"advanced.{key}" in APP_JS or f'"{key}"' in APP_JS
        # All but max_model_len/trust_remote_code are in the persisted set.
        assert "kv_cache_dtype" in _OVERRIDE_KEYS
        assert "extra_vllm_args" in _OVERRIDE_KEYS

    def test_kv_dtype_offers_auto_for_vision_models(self):
        """fp8 KV corrupts vision models on GB10 — the option has to exist."""
        assert 'value="auto"' in INDEX
        assert "vision" in INDEX

    def test_advanced_is_collapsed_by_default(self):
        """Nine extra fields must not push LAUNCH off the panel."""
        assert '<details class="launch-advanced"' in INDEX
        assert '<details class="launch-advanced" id="launch-advanced" open' not in INDEX


class TestSecuritySection:
    def test_nav_entry_exists(self):
        assert 'data-section="security"' in INDEX

    def test_renderer_is_wired(self):
        assert "case 'security'" in APP_JS
        assert "renderConfigSecurity()" in APP_JS

    def test_it_says_auth_is_off_by_default(self):
        """The single most important fact about a shared cluster."""
        assert "off by default" in APP_JS

    def test_key_is_shown_once_in_a_blocking_panel(self):
        assert "_showOneTimeKey" in APP_JS
        assert "onetime-key-backdrop" in APP_JS
        assert ".onetime-key-backdrop" in STYLE

    def test_copy_falls_back_without_a_secure_context(self):
        """A LAN node on plain http has no navigator.clipboard."""
        assert "navigator.clipboard" in APP_JS
        assert "Ctrl+C" in APP_JS

    def test_revocation_is_offered_per_key(self):
        assert "data-revoke" in APP_JS
        assert "/api/auth/keys/" in APP_JS


class TestAuthApiSupportsTheUi:
    def _status(self, cfg):
        from ainode.auth.api_routes import handle_auth_status

        class _Req:
            app = {"auth_config": cfg}

        return json.loads(asyncio.run(handle_auth_status(_Req())).body)

    def test_status_exposes_ids_so_keys_can_be_revoked(self, tmp_path, monkeypatch):
        from ainode.auth.middleware import AuthConfig

        monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
        cfg = AuthConfig()
        cfg.generate_key(name="anna")
        cfg.generate_key(name="bob")
        body = self._status(cfg)
        assert body["key_count"] == 2
        assert {k["name"] for k in body["keys"]} == {"anna", "bob"}
        assert all(k["id"] for k in body["keys"])

    def test_status_never_leaks_the_key_or_its_hash(self, tmp_path, monkeypatch):
        from ainode.auth.middleware import AuthConfig

        monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
        cfg = AuthConfig()
        created = cfg.generate_key(name="anna")
        body = self._status(cfg)
        blob = json.dumps(body)
        assert created["key"] not in blob
        assert "key_hash" not in blob

    def test_keys_carry_a_creation_time(self, tmp_path, monkeypatch):
        from ainode.auth.middleware import AuthConfig

        monkeypatch.setattr("ainode.auth.middleware.AUTH_FILE", tmp_path / "auth.json")
        cfg = AuthConfig()
        cfg.generate_key()
        assert self._status(cfg)["keys"][0]["created_at"] > 0


class TestEmbeddingsSurviveARestart:
    """They are in-process, so a restart drops them and a RAG pipeline breaks
    silently. LLM instances were already replayed; these were not."""

    def _manager(self, tmp_path, monkeypatch):
        from ainode.embeddings.manager import EmbeddingManager

        monkeypatch.setattr("ainode.core.config.AINODE_HOME", tmp_path)
        return EmbeddingManager()

    def test_manifest_round_trips(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["nomic-ai/nomic-embed-text-v1.5"] = object()
        m.save_manifest()
        assert m.load_manifest() == ["nomic-ai/nomic-embed-text-v1.5"]

    def test_missing_manifest_is_empty_not_an_error(self, tmp_path, monkeypatch):
        assert self._manager(tmp_path, monkeypatch).load_manifest() == []

    def test_unload_is_recorded_so_it_stays_unloaded(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["a"] = object()
        m._models["b"] = object()
        m.save_manifest()
        m._models.pop("a")
        m.save_manifest()
        assert m.load_manifest() == ["b"]

    def test_replay_skips_what_is_already_loaded(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["a"] = object()
        m.save_manifest()
        calls = []
        monkeypatch.setattr(type(m), "load", lambda self, mid, force=False: calls.append(mid))
        m.replay()
        assert calls == []

    def test_one_broken_model_does_not_block_the_others(self, tmp_path, monkeypatch):
        m = self._manager(tmp_path, monkeypatch)
        m._models["broken"] = object()
        m._models["good"] = object()
        m.save_manifest()
        m._models.clear()
        loaded = []

        def _load(self, mid, force=False):
            if mid == "broken":
                raise RuntimeError("gone from the hub")
            loaded.append(mid)

        monkeypatch.setattr(type(m), "load", _load)
        m.replay()          # must not raise
        assert loaded == ["good"]


class TestLauncherIsInTheImage:
    """EugrBackend shells out to /opt/spark-vllm-docker/launch-cluster.sh. It
    was present when this image was FROM ainode-base; since the split it is
    not, and every launch failed with "eugr launcher missing"."""

    DOCKERFILE = (Path(__file__).resolve().parent.parent
                  / "scripts" / "Dockerfile.ainode").read_text()
    BUILD_BASE = (Path(__file__).resolve().parent.parent
                  / "scripts" / "build-base-image.sh").read_text()

    def test_the_dockerfile_fetches_it(self):
        assert "/opt/spark-vllm-docker" in self.DOCKERFILE
        assert "launch-cluster.sh" in self.DOCKERFILE

    def test_the_fetch_is_verified_not_assumed(self):
        """A silent 404 would leave the same missing-launcher failure."""
        assert "test -x /opt/spark-vllm-docker/launch-cluster.sh" in self.DOCKERFILE

    def test_the_pin_matches_the_engine_build(self):
        """Launcher and engine hand each other a .env and a launch script, so a
        drifted pin is a wrong-shape contract rather than a clean failure."""
        import re

        dockerfile_sha = re.search(r"ARG EUGR_COMMIT=([0-9a-f]{40})", self.DOCKERFILE)
        base_sha = re.search(r'EUGR_COMMIT="\$\{EUGR_COMMIT:-([0-9a-f]{40})\}"',
                             self.BUILD_BASE)
        assert dockerfile_sha and base_sha
        assert dockerfile_sha.group(1) == base_sha.group(1)
