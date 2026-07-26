"""
hook.py
-------
The entire integration surface for sim/main.py. Add exactly two lines
right after the solver module is loaded (sim/main.py, ~line 245):

    _solver_module = importlib.import_module(_SOLVER_MODULE_NAME)
    from benchmark.hook import maybe_instrument                              # <-- add
    _solver_module.autopilot = maybe_instrument(                             # <-- add
        _solver_module.autopilot, solver_name=_SOLVER_MODULE_NAME,
        total_gates=len(ACTIVE_COURSE),
    )                                                                        # <-- add

That's it. No other file in sim/ or solver/ needs to change, and the
call site at line ~420 (`_solver_module.autopilot(solver_update)`)
doesn't need touching either, since we've replaced the attribute the
module looks up on that line, not the call itself.

Why this location and not the call site: sim/main.py is launched via
the `elodin` CLI (`elodin run sim/main.py`), not plain `python`, so an
external script that imports sim.main and calls some run()/main()
can't reliably stand in for that. Patching the attribute right after
import is the one integration point that's launcher-agnostic and
therefore robust regardless of how elodin invokes this file.

Zero risk to teammates who aren't benchmarking: `maybe_instrument`
is a no-op unless the BENCHMARK env var is set, so nobody else's run
is affected, timed, or slowed down.

Usage:
    BENCHMARK=1 RACE_SOLVER=solver.baseline elodin run sim/main.py

Results are exported to benchmark_results/<solver>__<run_id>/ via
`atexit`, so they're saved even if you stop the editor early with
Ctrl+C rather than waiting for MAX_TICKS to complete.
"""

from __future__ import annotations

import atexit
import os
from typing import Callable, Optional

from .benchmark import BenchmarkRun
from .logger import export_run

_active_run: Optional[BenchmarkRun] = None
_exported = False


def _export_once() -> None:
    global _exported
    if _active_run is None or _exported:
        return
    _exported = True
    try:
        out_dir = export_run(_active_run)
        print(f"[BENCHMARK] results written to {out_dir}")
    except Exception as e:  # never let export crash the sim on shutdown
        print(f"[BENCHMARK] export failed: {e}")


def maybe_instrument(
    autopilot_fn: Callable,
    solver_name: str,
    total_gates: Optional[int] = None,
    notes: str = "",
) -> Callable:
    """Return an instrumented autopilot() if BENCHMARK env var is truthy;
    otherwise return autopilot_fn completely unchanged."""
    global _active_run
    if not os.environ.get("BENCHMARK"):
        return autopilot_fn

    _active_run = BenchmarkRun(solver_name=solver_name, total_gates=total_gates, notes=notes)
    atexit.register(_export_once)
    print(f"[BENCHMARK] recording enabled for solver={solver_name!r} total_gates={total_gates}")
    return _active_run.instrument(autopilot_fn)


def current_run() -> Optional[BenchmarkRun]:
    """Exposed in case sim/main.py's own completion block (the
    'Simulation complete!' print at MAX_TICKS) wants to trigger an
    immediate export rather than waiting for atexit."""
    return _active_run