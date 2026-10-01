"""After the build and the model downloads, nothing is fetched or reported
online on its own.

Asked for: "hole bitte alle abhängigkeiten nach ainode so das keine inhalte
wie schriftarten etc online geholt werden müssen". Found: the dashboard loaded
Inter and JetBrains Mono from Google Fonts on every page view, every vLLM
engine ran its usage reporter (``_report_usage_worker``), and the embedding
model asked the Hub for a newer revision on every load.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from ainode.core.config import NodeConfig
from ainode.engine.no_telemetry import NO_TELEMETRY_ENV

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"


class TestTheDashboardLoadsNothingFromOutside:
    @pytest.mark.parametrize("page", sorted((WEB / "templates").glob("*.html")),
                             ids=lambda p: p.name)
    def test_no_external_stylesheet_script_font_or_image(self, page):
        html = page.read_text()
        # Links a person clicks are fine; anything the browser loads is not.
        loaded = re.findall(r'<(?:link|script|img|iframe|source)\b[^>]*(?:href|src)="(https?://[^"]+)"',
                            html)
        assert loaded == [], loaded
        assert "fonts.googleapis" not in html and "fonts.gstatic" not in html

    def test_no_stylesheet_or_script_imports_from_outside(self):
        for path in list((WEB / "static").rglob("*.css")) + list((WEB / "static").rglob("*.js")):
            text = path.read_text()
            assert not re.search(r'@import\s+(url\()?["\']?https?://', text), path
            assert not re.search(r'url\(\s*["\']?https?://', text), path
            assert not re.search(r'import\s+[^;]*from\s+["\']https?://', text), path

    def test_the_fonts_are_here_with_their_licences(self):
        fonts = WEB / "static" / "fonts"
        css = (fonts / "fonts.css").read_text()
        for name in re.findall(r"url\(\./([^)]+)\)", css):
            assert (fonts / name).is_file(), name
        assert "font-family: 'Inter'" in css and "font-family: 'JetBrains Mono'" in css
        assert (fonts / "OFL-Inter.txt").is_file() and (fonts / "OFL-JetBrainsMono.txt").is_file()

    def test_every_page_uses_them(self):
        for page in (WEB / "templates").glob("*.html"):
            if "font-family" in page.read_text() or page.name in ("index.html", "login.html",
                                                                  "onboarding.html"):
                assert "/static/fonts/fonts.css" in page.read_text(), page.name


class TestNoEngineReportsHome:
    def test_eugr_engines_get_it_before_the_recipe(self):
        from ainode.engine.backends.eugr import EugrBackend

        config = NodeConfig(node_id="n1", model="org/m", models_dir="/models",
                            extra_env={"DO_NOT_TRACK": "0"})
        args = EugrBackend(config)._launcher_env()["VLLM_SPARK_EXTRA_DOCKER_ARGS"]
        for key in NO_TELEMETRY_ENV:
            assert f"-e {key}=1" in args
        # The recipe's (here, the operator's) value comes later and wins.
        assert args.index("DO_NOT_TRACK=1") < args.index("DO_NOT_TRACK=0")

    def test_nvidia_engines_too(self):
        from ainode.engine.backends.nvidia import NvidiaBackend

        backend = NvidiaBackend.__new__(NvidiaBackend)
        backend.config = NodeConfig(extra_env={"HF_HUB_DISABLE_TELEMETRY": "0"})
        env = backend._engine_env({"NCCL_X": "1"})
        assert env["VLLM_NO_USAGE_STATS"] == "1" and env["NCCL_X"] == "1"
        assert env["HF_HUB_DISABLE_TELEMETRY"] == "0"

    def test_and_ainodes_own_process(self, monkeypatch):
        from ainode.engine.no_telemetry import apply_to_process

        for key in NO_TELEMETRY_ENV:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("DO_NOT_TRACK", "0")
        apply_to_process()
        assert os.environ["VLLM_NO_USAGE_STATS"] == "1"
        assert os.environ["DO_NOT_TRACK"] == "0"


class TestTheEmbeddingModelLoadsFromDisk:
    def test_local_first_then_the_hub(self, tmp_path):
        from ainode.embeddings.manager import EmbeddingManager

        calls = []

        class _ST:
            def __init__(self, model_id, cache_folder=None, local_files_only=False):
                calls.append(local_files_only)
                if local_files_only:
                    raise OSError("not cached")

            def get_sentence_embedding_dimension(self):
                return 8

        manager = EmbeddingManager(models_dir=str(tmp_path))
        manager._resolve_SentenceTransformer = lambda: _ST
        manager.load("org/emb")
        assert calls == [True, False]

    def test_a_cached_model_never_goes_to_the_hub(self, tmp_path):
        from ainode.embeddings.manager import EmbeddingManager

        calls = []

        class _ST:
            def __init__(self, model_id, cache_folder=None, local_files_only=False):
                calls.append(local_files_only)

            def get_sentence_embedding_dimension(self):
                return 8

        manager = EmbeddingManager(models_dir=str(tmp_path))
        manager._resolve_SentenceTransformer = lambda: _ST
        manager.load("org/emb")
        assert calls == [True]
