"""Tests for ainode.core.gpu."""

from unittest.mock import patch
from ainode.core.gpu import GPUInfo, detect_gpu, gpu_summary


def _make_gpu(**overrides):
    defaults = dict(
        name="NVIDIA GB10",
        memory_total_mb=131072,
        memory_free_mb=120000,
        cuda_version="12.8",
        driver_version="570.86",
        compute_capability="10.0",
        unified_memory=True,
    )
    defaults.update(overrides)
    return GPUInfo(**defaults)


def test_gpu_info_fields():
    gpu = _make_gpu()
    assert gpu.name == "NVIDIA GB10"
    assert gpu.unified_memory is True
    assert gpu.memory_total_mb == 131072


def test_detect_gpu_no_pynvml():
    """Returns None when pynvml is not available."""
    with patch.dict("sys.modules", {"pynvml": None}):
        result = detect_gpu()
        assert result is None


def test_gpu_summary_no_gpu():
    with patch("ainode.core.gpu.detect_gpu", return_value=None):
        assert gpu_summary() == "No NVIDIA GPU detected"


def test_gpu_summary_with_gpu():
    gpu = _make_gpu()
    with patch("ainode.core.gpu.detect_gpu", return_value=gpu):
        summary = gpu_summary()
        assert "NVIDIA GB10" in summary
        # 131072 MiB is 137 decimal GB. Real GB10 hardware reports about
        # 122070 MiB for its 128 GB, and the summary is decimal now — so the
        # fixture above is a binary-flavoured number, not a Spark. What this
        # asserts is that the conversion is decimal, which is the unit every
        # model size in this program is in. See ainode/core/units.py.
        assert "137 GB" in summary
        assert "unified memory" in summary
