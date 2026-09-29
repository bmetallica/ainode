"""FP8 storage for image models (image_dtype "fp8").

Asked for as "gibt es eine besser quantisierte version dieses modells welche
bei uns läuft? ca. 44gb vram bedarf wirkt mir doch recht viel" — about
Qwen-Image-2.1, which has no pre-quantized diffusers version. The image server
now stores the transformer and text encoder in FP8 and computes in bfloat16;
these tests hold the planner, the gate and the launch to the same figure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ainode.core.config import NodeConfig


def _pipeline(tmp_path: Path) -> Path:
    """A hub-shaped pipeline: 1000 bytes each of transformer and text
    encoder, 200 of VAE."""
    root = tmp_path / "models--org--img"
    snap = root / "snapshots" / "abc"
    for sub, size in (("transformer", 1000), ("text_encoder", 1000), ("vae", 200)):
        (snap / sub).mkdir(parents=True)
        (snap / sub / "w.safetensors").write_bytes(b"x" * size)
    (snap / "model_index.json").write_text("{}")
    return root


class _Manager:
    def __init__(self, root):
        self.root = root

    def model_dirs_for_repo(self, model):
        return [self.root]


class TestTheWeights:
    def test_bfloat16_counts_the_disk(self, tmp_path):
        from ainode.planner.api_routes import _image_weights_gb

        assert _image_weights_gb(_Manager(_pipeline(tmp_path)), "org/img") * 1e9 \
            == pytest.approx(2200)

    def test_fp8_halves_the_transformer_and_text_encoder_not_the_vae(self, tmp_path):
        from ainode.planner.api_routes import FP8_KEPT_FRACTION, _image_weights_gb

        got = _image_weights_gb(_Manager(_pipeline(tmp_path)), "org/img", "fp8") * 1e9
        assert got == pytest.approx(200 + 2000 * FP8_KEPT_FRACTION)

    def test_the_planner_and_the_server_cast_the_same_components(self):
        from ainode.engine import diffusers_server
        from ainode.planner import api_routes

        assert api_routes._FP8_COMPONENTS.pattern == diffusers_server._FP8_COMPONENTS.pattern
        rule = re.compile(api_routes._FP8_COMPONENTS.pattern)
        assert all(rule.match(n) for n in ("transformer", "text_encoder", "text_encoder_2", "unet"))
        assert not any(rule.match(n) for n in ("vae", "scheduler", "tokenizer"))


class TestTheLaunch:
    def test_fp8_is_accepted(self):
        from ainode.models.api_routes import parse_load_overrides

        overrides, error = parse_load_overrides({"image_dtype": "fp8"})
        assert error is None and overrides["image_dtype"] == "fp8"

    def test_anything_else_is_still_refused(self):
        from ainode.models.api_routes import parse_load_overrides

        _, error = parse_load_overrides({"image_dtype": "int4"})
        assert error is not None and error.status == 400

    def test_it_reaches_the_server_command_line(self):
        from ainode.engine.backends.diffusers import _dtype

        assert _dtype(NodeConfig(image_dtype="fp8")) == "fp8"

    def test_the_measurement_records_it(self):
        from ainode.measure.recorder import _launch_of

        launch = _launch_of(None, NodeConfig(engine_backend="diffusers", image_dtype="fp8"))
        assert launch["image_dtype"] == "fp8"


class TestThePlannersUseIt:
    def test_the_household_planner_asks_for_fp8_weights(self, monkeypatch):
        from ainode.planner import api_routes, household

        asked = []
        monkeypatch.setattr(api_routes, "_image_weights_gb",
                            lambda manager, model, dtype="": asked.append(dtype) or 17.0)
        item = household.Item(id="i", model="org/img", kind="image", node_ids=["n"])
        info = household._resolve_image({"model_manager": object()},
                                         {"image_dtype": "fp8", "max_image_size": 1024}, item)
        assert asked == ["fp8"] and info["fp8"] is True
        assert item.fixed_gb == pytest.approx(17.0 + 2.5 + 4.0)

    def test_the_gate_takes_the_launch_settings_over_the_nodes(self, monkeypatch):
        from ainode.safety import admission

        asked = []
        monkeypatch.setattr("ainode.planner.api_routes._image_weights_gb",
                            lambda manager, model, dtype="": asked.append(dtype) or 0.0)
        monkeypatch.setattr("ainode.planner.api_routes._is_image", lambda *a: True)
        monkeypatch.setattr("ainode.planner.api_routes._recipe", lambda app, model: None)
        monkeypatch.setattr("ainode.planner.facts.local_facts", lambda manager, model: None)
        app = {"config": NodeConfig(image_dtype="bfloat16")}
        assert admission._image_says(app, object(), "org/img", None,
                                     {"image_dtype": "fp8"}) == ""
        assert asked == ["fp8"]

    def test_a_bfloat16_measurement_does_not_stand_for_an_fp8_instance(self, tmp_path, monkeypatch):
        from ainode.measure.store import Measurement, MeasurementStore
        from ainode.models import api_routes

        store = MeasurementStore(tmp_path / "m.json")
        store._write({"org/img": Measurement(model="org/img", launches=1, last_ok=1.0,
                                             memory_gb=33.6,
                                             launch={"image_dtype": "bfloat16"})})
        monkeypatch.setattr("ainode.planner.api_routes._image_weights_gb",
                            lambda manager, model, dtype="": 18.0 if dtype == "fp8" else 33.0)
        collector = type("C", (), {"get_gpu_metrics":
                                   lambda self: {"memory_total_mb": 124650}})()
        app = {"measurement_store": store, "metrics_collector": collector,
               "model_manager": object()}
        inst = type("I", (), {})()
        inst.record = type("R", (), {"model": "org/img"})()
        inst.backend = type("B", (), {"config": NodeConfig(
            engine_backend="diffusers", image_dtype="fp8", max_image_size=1024)})()
        assert api_routes._reserved_share(app, inst) == pytest.approx((18.0 + 6.5) / 130.7, abs=0.01)
        inst.backend.config.image_dtype = "bfloat16"
        assert api_routes._reserved_share(app, inst) == pytest.approx(33.6 / 130.7, abs=0.01)
