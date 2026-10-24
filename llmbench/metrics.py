"""Timing, statistics and environment helpers."""

from __future__ import annotations

import math
import platform
import statistics
import time
from contextlib import contextmanager
from importlib import metadata


def percentile(values, q):
    """Linear-interpolated percentile (q in 0..100), same method as numpy's default."""
    if not values:
        raise ValueError("percentile of empty sequence")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values):
    """Summary statistics for a list of measurements."""
    values = list(values)
    if not values:
        return None
    return {
        "n": len(values),
        "median": percentile(values, 50),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def _cuda_sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def timed(out: dict, key: str = "elapsed"):
    """Wall-clock timer that synchronizes the GPU before and after the block.

    Without the synchronize calls, CUDA's async execution makes timings meaningless.
    """
    _cuda_sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _cuda_sync()
        out[key] = time.perf_counter() - start


def mb(num_bytes):
    return None if num_bytes is None else num_bytes / 2**20


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def env_info():
    """Hardware and library versions, recorded with every result for reproducibility."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "accelerate": _version("accelerate"),
        "gpu": None,
        "gpu_total_mb": None,
        "cuda": None,
    }
    try:
        import torch

        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_total_mb"] = round(props.total_memory / 2**20)
    except ImportError:
        pass
    return info
