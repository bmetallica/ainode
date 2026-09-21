"""The image-generation server, as it runs inside the engine container.

This file is not imported by AINode. The backend copies it next to its launch
script and the container executes it — the same arrangement the eugr path uses
for its launch script, and for the same reason: changing how a model is served
should not mean rebuilding a twenty-gigabyte image.

It therefore imports nothing from ``ainode``. Everything it needs arrives on
the command line, and it must run under whatever Python the engine image has.

The API is OpenAI's images endpoint, because every client already speaks it
and because AINode's proxy can then route it exactly like chat completions.
Everything diffusion-specific that OpenAI has no field for — steps, guidance,
seed, negative prompt — is accepted as an extra key and ignored when absent.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ainode.images")

STATE = {
    "ready": False,
    "error": "",
    "model": "",
    "pipeline": None,
    "images": 0,
    "seconds": 0.0,
    "steps": 0,
    "running": 0,
    "max_pixels": 0,
    "default_steps": 20,
    "default_size": "1024x1024",
}

#: One at a time. A diffusion run holds its activations for the whole of it,
#: and two concurrent runs on a unified-memory node do not halve the time,
#: they double the peak — which is the thing that takes the node down. The
#: queue is the HTTP server's; this only stops them overlapping in the GPU.
GPU_LOCK = threading.Lock()


def _parse_size(text: str) -> tuple:
    try:
        width, height = str(text).lower().split("x")
        return int(width), int(height)
    except Exception:
        raise ValueError(f"size must look like 1024x1024, not {text!r}")


def load_pipeline(path: str, dtype_name: str) -> None:
    """Load once, in a thread, so the HTTP port answers while it happens.

    The phrases below are the ones ainode/engine/load_phase.py already knows,
    so the load shows up on the instance card with no special casing — a
    launch is a launch whatever is being launched.
    """
    import torch
    from diffusers import DiffusionPipeline

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(dtype_name, torch.bfloat16)
    try:
        logger.info("Loading model weights from %s", path)
        pipeline = DiffusionPipeline.from_pretrained(path, torch_dtype=dtype)
        # .to("cuda") and nothing else. enable_model_cpu_offload() is the
        # standard advice for a machine with separate VRAM and is meaningless
        # here: on GB10 the CPU and the GPU share one physical pool, so
        # "offloading" moves nothing and pays for the copies.
        pipeline = pipeline.to("cuda")
        STATE["pipeline"] = pipeline
        STATE["ready"] = True
        logger.info("Application startup complete.")
    except Exception as exc:  # the card and the assistant read this line
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Engine core initialization failed: %s", STATE["error"])


def generate(body: dict) -> dict:
    pipeline = STATE["pipeline"]
    if pipeline is None:
        raise RuntimeError(STATE["error"] or "the model is still loading")

    prompt = body.get("prompt")
    if not prompt:
        raise ValueError("prompt is required")
    width, height = _parse_size(body.get("size") or STATE["default_size"])

    # The counterpart to --max-model-len: the one knob that stops a single
    # request taking the node down. A diffusion run's peak is its activations
    # and the VAE decode, and both grow with the square of the edge.
    limit = STATE["max_pixels"]
    if limit and width * height > limit:
        raise ValueError(
            f"{width}x{height} exceeds this instance's limit of {limit} pixels "
            f"({int(limit ** 0.5)}x{int(limit ** 0.5)} square). Raise "
            f"max_image_size when loading the model, if the node has room.")

    steps = int(body.get("steps") or body.get("num_inference_steps")
                or STATE["default_steps"])
    count = max(1, min(4, int(body.get("n") or 1)))

    kwargs = {"prompt": prompt, "width": width, "height": height,
              "num_inference_steps": steps, "num_images_per_prompt": count}
    if body.get("negative_prompt"):
        kwargs["negative_prompt"] = body["negative_prompt"]
    if body.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = float(body["guidance_scale"])
    seed = body.get("seed")
    if seed is not None:
        import torch

        kwargs["generator"] = torch.Generator("cuda").manual_seed(int(seed))

    started = time.time()
    STATE["running"] += 1
    try:
        with GPU_LOCK:
            result = pipeline(**kwargs)
    finally:
        STATE["running"] -= 1
    elapsed = time.time() - started

    STATE["images"] += count
    STATE["seconds"] += elapsed
    STATE["steps"] += steps * count

    data = []
    for image in result.images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data.append({"b64_json": base64.b64encode(buffer.getvalue()).decode()})
    return {"created": int(started), "data": data,
            "model": STATE["model"],
            "ainode": {"seconds": round(elapsed, 2), "steps": steps,
                       "size": f"{width}x{height}"}}


def metrics() -> str:
    """Prometheus text, named so AINode's scraper can distil it like vLLM's."""
    images = STATE["images"]
    seconds = STATE["seconds"]
    lines = [
        "# TYPE ainode:images_generated_total counter",
        f"ainode:images_generated_total {images}",
        "# TYPE ainode:image_seconds_total counter",
        f"ainode:image_seconds_total {round(seconds, 3)}",
        "# TYPE ainode:image_steps_total counter",
        f"ainode:image_steps_total {STATE['steps']}",
        "# TYPE ainode:requests_running gauge",
        f"ainode:requests_running {STATE['running']}",
    ]
    if images and seconds:
        lines += ["# TYPE ainode:seconds_per_image gauge",
                  f"ainode:seconds_per_image {round(seconds / images, 3)}",
                  "# TYPE ainode:steps_per_second gauge",
                  f"ainode:steps_per_second {round(STATE['steps'] / seconds, 2)}"]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # the server's own logger, not stderr
        logger.info("%s %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload, content_type="application/json"):
        body = (payload if isinstance(payload, bytes)
                else json.dumps(payload).encode())
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/metrics"):
            return self._send(200, metrics().encode(), "text/plain")
        if self.path.startswith("/health"):
            return self._send(200 if STATE["ready"] else 503, {
                "ready": STATE["ready"], "error": STATE["error"],
                "model": STATE["model"]})
        if self.path.startswith("/v1/models"):
            # The readiness probe AINode already uses for every engine. It
            # must answer 200 only once the model can actually serve, or the
            # instance is marked ready while the weights are still loading.
            if not STATE["ready"]:
                return self._send(503, {"error": "still loading"})
            return self._send(200, {"object": "list", "data": [
                {"id": STATE["model"], "object": "model", "owned_by": "ainode"}]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/images/generations"):
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, {"error": "invalid JSON"})
        try:
            return self._send(200, generate(body))
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("generation failed")
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-image-size", type=int, default=1536,
                        help="longest square edge this instance will accept")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", default="1024x1024")
    args = parser.parse_args()

    STATE["model"] = args.served_model_name or args.model_path
    STATE["max_pixels"] = max(0, args.max_image_size) ** 2
    STATE["default_steps"] = args.steps
    STATE["default_size"] = args.size

    # The port answers before the weights are in, so the load is visible as a
    # load rather than as a node that is not listening.
    threading.Thread(target=load_pipeline,
                     args=(args.model_path, args.dtype), daemon=True).start()
    logger.info("Starting image server on port %s for %s", args.port,
                STATE["model"])
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
