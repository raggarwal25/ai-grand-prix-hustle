"""
logger.py
---------
Exports a completed BenchmarkRun to disk: a per-tick CSV, a JSON
summary, a JSON run manifest, and appends a one-line summary to
benchmark_history.csv.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .benchmark import BenchmarkRun

DEFAULT_RESULTS_DIR = Path("benchmark_results")


def _run_dir(run: BenchmarkRun, base_dir: Path) -> Path:
    name = f"{run.meta.solver_name}__{run.meta.run_id}"
    return base_dir / name


def export_csv(run: BenchmarkRun, path: Path) -> None:
    frames = run.frames

    path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        path.write_text("")
        return

    fieldnames = list(frames[0].as_dict().keys())

    # Union of keys across frames, in case `extra` differs per-frame.
    for f in frames[1:]:
        for k in f.as_dict().keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for f in frames:
            writer.writerow(f.as_dict())


def export_summary(run: BenchmarkRun, path: Path) -> None:
    if run.summary is None:
        run.finalize()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.summary.as_dict(), indent=2))


def export_manifest(run: BenchmarkRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(run.meta), indent=2))


def export_run(run: BenchmarkRun, base_dir: Optional[Path] = None) -> Path:
    """One-call convenience export. Returns the folder written to."""

    base_dir = Path(base_dir) if base_dir else DEFAULT_RESULTS_DIR

    if run.summary is None:
        run.finalize()

    summary = run.summary

    # Skip runs that never recorded telemetry.
    if summary.num_ticks == 0:
        return base_dir

    out_dir = _run_dir(run, base_dir)

    export_csv(run, out_dir / "telemetry.csv")
    export_summary(run, out_dir / "summary.json")
    export_manifest(run, out_dir / "manifest.json")

    append_history(run, base_dir)

    return out_dir


def load_summary(path: Path) -> dict:
    """Used by compare.py to reload a previously-exported run's summary."""
    return json.loads(Path(path).read_text())


def append_history(run: BenchmarkRun, base_dir: Optional[Path] = None) -> None:
    """Append a one-line summary of this run to benchmark_history.csv."""

    base_dir = Path(base_dir) if base_dir else DEFAULT_RESULTS_DIR
    history_file = base_dir / "benchmark_history.csv"

    if run.summary is None:
        run.finalize()

    summary = run.summary

    # Skip empty/failed runs.
    if summary.num_ticks == 0:
        return

    row = {
        "timestamp": datetime.fromtimestamp(
            run.meta.finished_at_wall
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run.meta.run_id,
        "solver": run.meta.solver_name,
        "status": summary.race_status,
        "lap_time_s": summary.lap_time_s,
        "gates_passed": summary.gates_passed,
        "total_gates": summary.total_gates,
        "average_speed": summary.average_speed,
        "max_speed": summary.max_speed,
        "distance_travelled": summary.distance_travelled,
        "mean_solver_latency_s": summary.solver_latency["mean_s"],
    }

    history_file.parent.mkdir(parents=True, exist_ok=True)

    write_header = not history_file.exists()

    with history_file.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)