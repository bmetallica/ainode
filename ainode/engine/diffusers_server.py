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
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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
        STATE["error"] = _explain(exc, path)
        logger.error("Engine core initialization failed: %s", STATE["error"])
        # And then stop. A server that cannot load has nothing to serve, and
        # staying up means answering 503 for ever while the instance card
        # sits at "loading weights" — which is what this looked like from the
        # UI: a progress bar for a load that had died in its first second.
        # Exiting ends the log stream, and the backend reports a launch that
        # failed, with these lines as the evidence.
        logging.shutdown()
        os._exit(1)


#: diffusers instantiates the class named in model_index.json by looking it
#: up on its own module, so a checkpoint newer than the installed library
#: fails with an AttributeError naming a class nobody can find.
_MISSING_CLASS = re.compile(
    r"module diffusers has no attribute (\w+)", re.IGNORECASE)


def _explain(exc: Exception, path: str) -> str:
    """The exception, plus what it means when it is the version one."""
    text = f"{type(exc).__name__}: {exc}"
    match = _MISSING_CLASS.search(str(exc))
    if not match:
        return text
    try:
        import diffusers

        installed = getattr(diffusers, "__version__", "?")
    except Exception:  # pragma: no cover - diffusers is imported above
        installed = "?"
    wanted = ""
    try:
        index = json.loads(
            (Path(path) / "model_index.json").read_text())
        wanted = str(index.get("_diffusers_version") or "")
    except Exception:
        pass
    built_with = f", and this checkpoint was written with {wanted}" if wanted else ""
    advice = (
        "Rebuild the engine image against a diffusers that has it:\n"
        "    DIFFUSERS_REF=\"git+https://github.com/huggingface/diffusers\" "
        "scripts/build-diffusers-image.sh")
    if _at_least(installed, wanted):
        # Newer than the checkpoint and still missing the class. Telling
        # anyone to upgrade further would send them around the same loop: a
        # class that is in no release is in no newer release either.
        advice = (
            "The installed version is already newer than the one the "
            "checkpoint names, so upgrading to another RELEASE will not "
            "help — this class is not in one. A checkpoint published with "
            "its architecture usually needs diffusers from git, and the "
            "model card says which:\n"
            "    DIFFUSERS_REF=\"git+https://github.com/huggingface/diffusers\" "
            "scripts/build-diffusers-image.sh")
    return (
        f"{text} — this engine image has diffusers {installed}{built_with}. "
        f"{match.group(1)} does not exist in the installed version, so the "
        f"pipeline named in model_index.json cannot be instantiated. This is "
        f"the image, not the model or the launch. {advice}"
    )


def _at_least(installed: str, wanted: str) -> bool:
    """True when ``installed`` is not older than ``wanted``.

    Compared on the numeric head only: "0.40.0" against "0.37.0.dev0" is a
    comparison of (0, 40, 0) and (0, 37, 0), and the dev suffix says which
    side of a release it sits on, not which release.
    """
    def _parts(value):
        head = re.match(r"(\d+(?:\.\d+)*)", str(value or ""))
        return tuple(int(n) for n in head.group(1).split(".")) if head else ()

    left, right = _parts(installed), _parts(wanted)
    if not left or not right:
        return False
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


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
