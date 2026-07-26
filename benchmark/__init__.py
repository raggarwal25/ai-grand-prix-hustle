"""
Public API for the benchmark package.

Typical usage:

    from benchmark import BenchmarkRun, maybe_instrument
"""
from .benchmark import BenchmarkRun, RunMetadata
from .telemetry import TelemetryFrame, TelemetryCollector
from .metrics import compute_summary, RunSummary
from .logger import export_run, export_csv, export_summary, export_manifest, load_summary
from .hook import maybe_instrument, current_run

__all__ = [
    "BenchmarkRun",
    "RunMetadata",
    "TelemetryFrame",
    "TelemetryCollector",
    "compute_summary",
    "RunSummary",
    "export_run",
    "export_csv",
    "export_summary",
    "export_manifest",
    "load_summary",
    "maybe_instrument",
    "current_run",
]