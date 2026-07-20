"""
benchmark.py
------------
`BenchmarkRun` represents a single evaluation of one solver, from arm
to finish (or crash/timeout). It owns a TelemetryCollector and knows
how to produce an instrumented version of `autopilot()`.

Two integration modes are supported:

1. RECOMMENDED — external wrapping (no repo files touched at all):
   Wrap `solver_module.autopilot` before handing control to
   `sim/main.py`. See scripts/run_benchmark.py for the full example:

       run = BenchmarkRun(solver_name="baseline")
       solver_module.autopilot = run.instrument(solver_module.autopilot)
       sim.main.run(...)
       summary = run.finalize()

2. FALLBACK — inline hook, if the sim loads/reloads the solver module
   in a way that defeats external monkeypatching (e.g. re-imports it
   fresh every tick). Add two lines inside sitl_post_step():

       rc_out = run.instrumented_autopilot(solver_update)   # instead of
       # rc_out = _solver_module.autopilot(solver_update)

   `run.instrumented_autopilot` is a bound method equivalent to
   `run.instrument(_solver_module.autopilot)` computed lazily on first
   call, so it also works if you only have access to the module at
   that point, not before the sim starts.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .telemetry import TelemetryCollector, TelemetryFrame
from .metrics import RunSummary, compute_summary

AutopilotFn = Callable[[Any], Any]


@dataclass
class RunMetadata:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    solver_name: str = "unknown"
    notes: str = ""
    total_gates: Optional[int] = None
    started_at_wall: float = field(default_factory=time.time)
    finished_at_wall: Optional[float] = None


class BenchmarkRun:
    """One benchmarking session for one solver."""

    def __init__(
        self,
        solver_name: str = "unknown",
        total_gates: Optional[int] = None,
        notes: str = "",
    ) -> None:
        self.meta = RunMetadata(solver_name=solver_name, total_gates=total_gates, notes=notes)
        self.collector = TelemetryCollector()
        self._summary: Optional[RunSummary] = None
        self._raw_autopilot: Optional[AutopilotFn] = None

    # ---- instrumentation -------------------------------------------------

    def instrument(self, autopilot_fn: AutopilotFn) -> AutopilotFn:
        """Return a wrapped autopilot() that transparently records
        telemetry + latency, then forwards the real RC command through
        unchanged. The solver's behavior and return value are never
        modified — this is a pure observer."""
        self._raw_autopilot = autopilot_fn

        def wrapped(update: Any) -> Any:
            t0 = time.perf_counter()
            rc_out = autopilot_fn(update)
            dt = time.perf_counter() - t0
            self.collector.collect(update, rc_out=rc_out, latency_s=dt)
            return rc_out

        return wrapped

    def instrumented_autopilot(self, update: Any) -> Any:
        """Convenience entry point for the inline-hook integration
        style (see module docstring, option 2). Requires that
        `bind_solver()` was called first with the real autopilot fn."""
        if self._raw_autopilot is None:
            raise RuntimeError(
                "instrumented_autopilot() called before bind_solver(). "
                "Call run.bind_solver(_solver_module.autopilot) once at "
                "startup, or use run.instrument(...) to wrap externally."
            )
        t0 = time.perf_counter()
        rc_out = self._raw_autopilot(update)
        dt = time.perf_counter() - t0
        self.collector.collect(update, rc_out=rc_out, latency_s=dt)
        return rc_out

    def bind_solver(self, autopilot_fn: AutopilotFn) -> None:
        self._raw_autopilot = autopilot_fn

    # ---- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "BenchmarkRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finalize()
        return False  # never swallow exceptions from the sim

    def finalize(self) -> RunSummary:
        self.meta.finished_at_wall = time.time()
        self._summary = compute_summary(self.collector.frames, total_gates=self.meta.total_gates)
        return self._summary

    @property
    def summary(self) -> Optional[RunSummary]:
        return self._summary

    @property
    def frames(self) -> list[TelemetryFrame]:
        return self.collector.frames