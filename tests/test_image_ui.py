"""The two places image generation becomes visible: the launch form and a
panel to use it from.

Everything else about it — the card, the guard, placement, profiles — is the
machinery that was already there, and is tested where that machinery lives.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "ainode" / "web"
APP_JS = (WEB / "static" / "js" / "app.js").read_text()
INDEX = (WEB / "templates" / "index.html").read_text()
CSS = (WEB / "static" / "css" / "style.css").read_text()


class TestTheLaunchForm:
    def test_an_image_model_is_marked_in_the_list(self):
        assert 'data-modality="' in APP_JS

    def test_its_own_fields_appear_only_for_one(self):
        assert 'id="launch-image-fields"' in INDEX
        assert "toggleImageFields" in APP_JS

    def test_the_kv_fields_are_hidden_for_it(self):
        # A diffusion run has no KV cache and no context length. Leaving them
        # on screen invites someone to set them and wonder why nothing
        # changed.
        block = APP_JS.split("toggleImageFields(model) {")[1].split("\n  },")[0]
        for field in ("launch-max-seqs", "launch-max-len", "launch-kv-dtype"):
            assert field in block, field

    def test_the_resolution_limit_is_offered_and_explained(self):
        assert 'id="launch-max-image"' in INDEX
        # Why it is the one that matters.
        assert "peak lands at the <strong>end</strong>" in INDEX

    def test_the_settings_reach_the_launch_body(self):
        assert "imageOverrides" in APP_JS
        launch = APP_JS.split("async launchInstance(")[1][:6000]
        assert "imageOverrides()" in launch


class TestThePanel:
    def test_there_is_a_view_for_it(self):
        assert 'data-view="images"' in INDEX
        assert 'id="view-images"' in INDEX
        assert "renderImages" in APP_JS

    def test_it_says_so_when_nothing_is_loaded(self):
        block = APP_JS.split("async renderImages() {")[1][:1200]
        assert "No image model is loaded" in block

    def test_it_lists_only_image_instances(self):
        # Read from the same instance list the cards use, so the two cannot
        # disagree about what is running.
        block = APP_JS.split("imageInstances() {")[1].split("\n  },")[0]
        assert "inst.kind === 'image'" in block

    def test_the_poll_does_not_wipe_a_half_typed_prompt(self):
        # The view re-renders every five seconds like everything else.
        block = APP_JS.split("async renderImages() {")[1][:1600]
        assert "_imagesDrawnFor" in block

    def test_it_shows_the_elapsed_time_while_it_waits(self):
        # A picture takes tens of seconds and the engine reports no progress,
        # so the honest thing to show is the clock.
        block = APP_JS.split("async generateImage() {")[1][:3000]
        assert "generating…" in block

    def test_it_posts_the_openai_shape(self):
        block = APP_JS.split("async generateImage() {")[1][:3000]
        assert "'/v1/images/generations'" in block
        assert "response_format: 'b64_json'" in block

    def test_the_diffusion_knobs_are_offered(self):
        block = APP_JS.split("async renderImages() {")[1][:3000]
        for field in ("image-size", "image-steps", "image-seed",
                      "image-negative"):
            assert field in block, field

    def test_the_gallery_is_bounded(self):
        # Forty pictures at a megabyte each is a tab that runs out of memory.
        block = APP_JS.split("showImages(data, request, seconds) {")[1]
        assert "figures[i].remove()" in block

    def test_a_result_carries_what_made_it(self):
        # A picture with no prompt, size, steps or seed beside it is a
        # picture nobody can reproduce.
        block = APP_JS.split("showImages(data, request, seconds) {")[1]
        assert "request.prompt" in block and "request.steps" in block
        assert "seed" in block

    def test_the_gallery_has_a_layout(self):
        assert ".image-gallery" in CSS
        assert ".image-card img" in CSS


class TestTelemetry:
    def test_the_image_engine_reports_on_the_same_topic(self):
        from ainode.telemetry.engine_metrics import distil

        out = distil(
            "ainode:images_generated_total 12\n"
            "ainode:image_seconds_total 496.0\n"
            "ainode:seconds_per_image 41.3\n"
            "ainode:steps_per_second 0.48\n"
            "ainode:requests_running 1\n")
        assert out["images_generated_total"] == 12
        assert out["requests_running"] == 1

    def test_rates_keep_their_decimals(self):
        # Rounding 0.48 steps a second to 0 reads as a stalled engine.
        from ainode.telemetry.engine_metrics import distil

        out = distil("ainode:steps_per_second 0.48\n"
                     "ainode:seconds_per_image 41.3\n")
        assert out["steps_per_second"] == 0.48
        assert out["seconds_per_image"] == 41.3

    def test_a_rate_of_zero_is_left_out(self):
        # The server only emits these once it has made something; a zero would
        # read as infinitely fast on a gauge.
        from ainode.telemetry.engine_metrics import distil

        out = distil("ainode:seconds_per_image 0\nainode:images_generated_total 0\n")
        assert "seconds_per_image" not in out

    def test_an_image_instance_is_labelled_in_the_models_payload(self):
        from ainode.core.config import NodeConfig
        from ainode.telemetry.payloads import build_payloads

        class _Record:
            model = "org/img"
            api_port = 8003
            status = "serving"
            peer_ips = []
            kind = "image"

        class _Instance:
            record = _Record()
            backend = type("B", (), {"config": None, "load_phase": "ready"})()

        class _Manager:
            def instances(self):
                return [_Instance()]

        class _Sampler:
            def sample(self):
                return {}

        payload = build_payloads(
            {"config": NodeConfig(node_id="n"), "instances": _Manager()},
            _Sampler())
        assert payload["models"]["loaded"][0]["kind"] == "image"

    def test_an_llm_carries_no_kind_at_all(self):
        # The default stays unlabelled so a dashboard from before image
        # generation reads exactly as it did.
        from ainode.core.config import NodeConfig
        from ainode.telemetry.payloads import build_payloads

        class _Record:
            model = "org/llm"
            api_port = 8000
            status = "serving"
            peer_ips = []
            kind = "llm"

        class _Instance:
            record = _Record()
            backend = type("B", (), {"config": None, "load_phase": "ready"})()

        class _Manager:
            def instances(self):
                return [_Instance()]

        class _Sampler:
            def sample(self):
                return {}

        payload = build_payloads(
            {"config": NodeConfig(node_id="n"), "instances": _Manager()},
            _Sampler())
        assert "kind" not in payload["models"]["loaded"][0]
