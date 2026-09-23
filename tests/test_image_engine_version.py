"""An image load that fails has to say so, and say what is wrong.

From the cluster, with the instance card stuck at 40%:

    2026-09-23 15:34:24,688 INFO Loading model weights from /models/...
    2026-09-23 15:34:24,688 ERROR Engine core initialization failed:
        AttributeError: module diffusers has no attribute QwenImage21Pipeline
    ... GET /v1/models 503 ... GET /v1/models 503 ... (for ever)

Two faults in three lines. The load died in its first second and the server
stayed up answering 503, so the UI showed a progress bar for something that
was never going to finish. And the error names a symptom: diffusers
instantiates the class from model_index.json by looking it up on its own
module, so a checkpoint newer than the installed library fails with a name
nobody can place. Qwen-Image-2.1 says `"_diffusers_version": "0.37.0.dev0"`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = (ROOT / "ainode" / "engine" / "diffusers_server.py").read_text()


class TestItStopsWhenItCannotLoad:
    def test_the_process_exits(self):
        # A server that cannot load has nothing to serve, and staying up
        # means answering 503 for ever while the card says "loading".
        block = SERVER.split("Engine core initialization failed")[1][:600]
        assert "os._exit(1)" in block

    def test_the_log_is_flushed_first(self):
        # os._exit skips buffers, and the error line is the evidence the
        # backend reports.
        block = SERVER.split("Engine core initialization failed")[1][:600]
        assert block.index("logging.shutdown()") < block.index("os._exit(1)")

    def test_the_readiness_probe_still_answers_503_while_loading(self):
        # Only a FAILED load exits; a slow one must stay up, or every load
        # would look like a failure.
        assert "503" in SERVER


class TestItExplainsAVersionMismatch:
    def _explain(self, message, tmp_path=None, version=""):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "diffusers_server_undertest",
            ROOT / "ainode" / "engine" / "diffusers_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path = str(tmp_path) if tmp_path else "/nowhere"
        if tmp_path and version:
            (tmp_path / "model_index.json").write_text(
                json.dumps({"_class_name": "QwenImage21Pipeline",
                            "_diffusers_version": version}))
        return module._explain(AttributeError(message), path)

    def test_a_missing_class_is_named_as_a_version_problem(self, tmp_path):
        out = self._explain("module diffusers has no attribute QwenImage21Pipeline",
                            tmp_path, "0.37.0.dev0")
        assert "QwenImage21Pipeline does not exist in the installed version" in out
        assert "0.37.0.dev0" in out

    def test_it_says_which_side_to_change(self):
        out = self._explain("module diffusers has no attribute QwenImage21Pipeline")
        assert "This is the image, not the model or the launch" in out
        assert "DIFFUSERS_REF" in out

    def test_an_unrelated_failure_is_passed_through(self):
        out = self._explain("CUDA out of memory")
        assert out == "AttributeError: CUDA out of memory"

    def test_a_checkpoint_with_no_index_still_explains_what_it_can(self):
        out = self._explain("module diffusers has no attribute Whatever")
        assert "Whatever does not exist" in out


class TestTheBuildCanPinDiffusers:
    DOCKERFILE = (ROOT / "scripts" / "Dockerfile.diffusers").read_text()
    BUILD = (ROOT / "scripts" / "build-diffusers-image.sh").read_text()

    def test_the_dockerfile_takes_a_ref(self):
        assert 'ARG DIFFUSERS_REF' in self.DOCKERFILE
        assert '"${DIFFUSERS_REF}"' in self.DOCKERFILE

    def test_the_default_is_a_release(self):
        # A git ref is for the checkpoint that needs one, not for everybody.
        assert 'ARG DIFFUSERS_REF="diffusers>=0.36"' in self.DOCKERFILE

    def test_the_script_passes_it_through(self):
        assert 'DIFFUSERS_REF=${DIFFUSERS_REF}' in self.BUILD

    def test_the_script_says_what_it_installed(self):
        assert 'diffusers: ${DIFFUSERS_REF}' in self.BUILD

    def test_it_documents_the_symptom(self):
        assert "QwenImage21Pipeline" in self.BUILD


class TestWhenTheInstalledOneIsAlreadyNewer:
    """Reported after rebuilding:

        this engine image has diffusers 0.40.0, and this checkpoint was
        written with 0.37.0.dev0. QwenImage21Pipeline does not exist in the
        installed version

    0.40 is newer than 0.37 and the class is still missing — because it is in
    no release at all. Qwen-Image-2.1's own card says
    `pip install git+https://github.com/huggingface/diffusers`. Telling
    anyone to upgrade further would send them around the same loop.
    """

    def _explain(self, tmp_path, installed, wanted):
        import importlib.util
        from unittest import mock

        spec = importlib.util.spec_from_file_location(
            "diffusers_server_versions",
            ROOT / "ainode" / "engine" / "diffusers_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_diffusers_version": wanted}))
        fake = type("m", (), {"__version__": installed})
        with mock.patch.dict("sys.modules", {"diffusers": fake}):
            return module._explain(
                AttributeError(
                    "module diffusers has no attribute QwenImage21Pipeline"),
                str(tmp_path))

    def test_it_does_not_send_you_to_another_release(self, tmp_path):
        out = self._explain(tmp_path, "0.40.0", "0.37.0.dev0")
        assert "already newer" in out
        assert "will not help" in out

    def test_it_names_the_install_the_card_names(self, tmp_path):
        out = self._explain(tmp_path, "0.40.0", "0.37.0.dev0")
        assert "git+https://github.com/huggingface/diffusers" in out

    def test_an_older_install_is_told_to_upgrade(self, tmp_path):
        out = self._explain(tmp_path, "0.36.2", "0.37.0.dev0")
        assert "already newer" not in out
        assert "DIFFUSERS_REF" in out

    def test_the_comparison_ignores_the_dev_suffix(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "diffusers_server_cmp",
            ROOT / "ainode" / "engine" / "diffusers_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._at_least("0.40.0", "0.37.0.dev0") is True
        assert module._at_least("0.37.0", "0.37.0.dev0") is True
        assert module._at_least("0.36.2", "0.37.0.dev0") is False
        assert module._at_least("1.0", "0.40.0") is True

    def test_an_unreadable_version_does_not_claim_to_know(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "diffusers_server_cmp2",
            ROOT / "ainode" / "engine" / "diffusers_server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module._at_least("", "0.37.0") is False
        assert module._at_least("0.40.0", "") is False
