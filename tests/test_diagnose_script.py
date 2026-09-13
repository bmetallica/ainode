"""The diagnostic report must be safe to paste and complete enough to act on.

Safe first: an HF token has already been pasted into a chat once in this
project's life. A report an operator has to audit line by line before sending
is a report nobody will send, so the redaction is part of the contract and is
tested against real-looking credentials.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diagnose.sh"


def _redact(text: str) -> str:
    """Run the script's own redact() over text, extracted at test time."""
    body = re.search(r"^redact\(\) \{.*?^\}", SCRIPT.read_text(), re.S | re.M)
    assert body, "redact() is no longer where the test expects it"
    result = subprocess.run(
        ["bash", "-c", f"{body.group(0)}\nredact"],
        input=text, capture_output=True, text=True, timeout=30,
    )
    return result.stdout


class TestRedaction:
    # Assembled from parts rather than written out. A literal that LOOKS like
    # a credential is one: GitHub's push protection rejected this file when it
    # first carried a real-shaped token — the system working as intended — and
    # a test fixture is not worth teaching anyone that pasting one is normal.
    FAKE = {
        "hf": "hf_" + "x" * 34,
        "hf_short": "hf_" + "y" * 20,
        "openai": "sk-proj-" + "z" * 24,
        "ngc": "nvapi-" + "q" * 24,
        "bearer": "eyJ" + "w" * 30 + ".payload",
        "cluster": "not-a-real-cluster-secret",
        "mqtt": "not-a-real-password",
    }

    @pytest.mark.parametrize("key,template", [
        ("hf", "Warning: token {} used"),
        ("hf_short", '"huggingface_token": "{}",'),
        ("cluster", '"cluster_secret": "{}",'),
        ("mqtt", '"mqtt_password": "{}",'),
        ("openai", "OPENAI_API_KEY={}"),
        ("ngc", "ngc key {}"),
        ("bearer", "Authorization: Bearer {}"),
        ("hf_short", "HUGGING_FACE_HUB_TOKEN={}"),
    ])
    def test_a_secret_does_not_survive(self, key, template):
        secret = self.FAKE[key]
        assert secret not in _redact(template.format(secret))

    def test_it_redacts_rather_than_deletes(self):
        # A blank where a token was leaves the reader guessing whether one is
        # configured at all.
        line = '"huggingface_token": "%s"' % self.FAKE["hf_short"]
        assert "REDACTED" in _redact(line)

    @pytest.mark.parametrize("line", [
        "vllm serve /models/unsloth--Qwen3.8-27B-NVFP4 --port 8000",
        "NCCL_IB_HCA=mlx5_0,mlx5_1",
        "RuntimeError: CUDA driver error: an illegal instruction was encountered",
        "192.168.1.2:5001/ainode:dev",
    ])
    def test_the_evidence_survives(self, line):
        # Over-redaction would be its own failure: these are the lines the
        # report exists to carry.
        assert _redact(line).strip() == line


class TestItIsReadOnly:
    @pytest.mark.parametrize("forbidden", [
        "docker stop", "docker rm", "docker kill", "systemctl restart",
        "systemctl stop", "rm -rf", "docker run",
    ])
    def test_it_never_changes_anything(self, forbidden):
        # An operator running a diagnostic on a half-broken cluster must not
        # have the diagnostic finish the job.
        for line in SCRIPT.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert forbidden not in stripped, stripped


class TestItCollectsWhatIsNeeded:
    @pytest.mark.parametrize("probe", [
        "/api/status",                 # version, phase, error
        "/api/cluster/resources",      # what the cluster thinks is running
        "/api/nodes",                  # per-instance status across nodes
        "ainode-*.sh",                 # what the engine was actually told
        "docker inspect vllm_node",    # mounts and environment
        "docker images",               # image ids per node
        "/root/.ainode/models",        # is the model on disk
        "config.json",
        "vllm.log",
        "distributed.log",
        "docker logs",
        "journalctl",
    ])
    def test_the_report_includes(self, probe):
        assert probe in SCRIPT.read_text()

    def test_it_can_reach_the_peers(self):
        text = SCRIPT.read_text()
        assert "--nodes" in text and "ssh -o BatchMode=yes" in text

    def test_log_sections_are_filtered_not_dumped(self):
        # vllm.log is 600 KB of throughput counters with the interesting lines
        # thousands of lines before the end.
        text = SCRIPT.read_text()
        assert "Avg prompt throughput" in text      # the noise it drops
        assert "init engine" in text                # the timing it keeps
        assert "Capturing CUDA graph" in text

    def test_a_failing_probe_does_not_end_the_report(self):
        text = SCRIPT.read_text()
        assert "set -uo pipefail" in text
        assert "set -euo pipefail" not in text


class TestItRuns:
    def test_it_completes_where_nothing_is_installed(self):
        # Every probe fails on a machine with no cluster; the report must
        # still be produced, because "everything failed" is itself the finding.
        result = subprocess.run(["bash", str(SCRIPT)], capture_output=True,
                                text=True, timeout=300)
        assert result.returncode == 0
        assert "end of report" in result.stdout
        assert result.stdout.count("=====") > 20

    def test_help_does_not_run_probes(self):
        result = subprocess.run(["bash", str(SCRIPT), "--help"],
                                capture_output=True, text=True, timeout=30)
        assert "Read-only" in result.stdout
        assert "=====" not in result.stdout
