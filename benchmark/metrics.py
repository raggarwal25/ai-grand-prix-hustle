"""
metrics.py
----------
Pure, stateless functions that turn a list of TelemetryFrame into
numbers. Nothing here touches files or holds run state, so these
functions are trivially unit-testable and reusable by compare.py.

Gate/lap metrics are reconstructed independently from the recorded
`last_gate_passed` stream (rather than trusting a single external
value), which doubles as a sanity check against sim/main.py's own
`race_course.print_summary` line for the same run.

There is currently no collision/crash signal in this sim (see
telemetry.py docstring), so no collision metrics are computed here —
we don't fabricate a number for something the sim doesn't expose yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import List, Optional, Sequence

from .telemetry import TelemetryFrame, Vec3


def _norm(v: Vec3) -> float:
    return sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def speeds(frames: Sequence[TelemetryFrame]) -> List[float]:
    """Linear ground speed per tick, m/s (norm of world_vel[3:6])."""
    return [_norm(f.linear_velocity) for f in frames]


def average_speed(frames: Sequence[TelemetryFrame]) -> float:
    s = speeds(frames)
    return mean(s) if s else 0.0


def max_speed(frames: Sequence[TelemetryFrame]) -> float:
    s = speeds(frames)
    return max(s) if s else 0.0


def distance_travelled(frames: Sequence[TelemetryFrame]) -> float:
    """Integrates path length from consecutive `position` samples
    (derived from world_pos[4:7]). Robust to uneven tick spacing."""
    if len(frames) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(frames, frames[1:]):
        dx = b.position[0] - a.position[0]
        dy = b.position[1] - a.position[1]
        dz = b.position[2] - a.position[2]
        total += sqrt(dx * dx + dy * dy + dz * dz)
    return total


def flight_duration(frames: Sequence[TelemetryFrame]) -> float:
    if not frames:
        return 0.0
    return frames[-1].t - frames[0].t


def gates_passed(frames: Sequence[TelemetryFrame]) -> int:
    """Matches sim/main.py's own accounting: n_passed = last_gate_passed + 1.
    Uses the max seen last_gate_passed across the run (it's monotonic
    non-decreasing in the sim, but we don't assume tick ordering)."""
    if not frames:
        return 0
    max_last_gate = max(f.last_gate_passed for f in frames)
    return max(0, max_last_gate + 1)


def gate_pass_times(frames: Sequence[TelemetryFrame]) -> dict:
    """Reconstructs {gate_index: t_first_seen_passed} purely from the
    recorded last_gate_passed stream, independent of sim/main.py's own
    _race_pass_times bookkeeping — useful as a cross-check."""
    times: dict = {}
    prev = -1
    for f in frames:
        if f.last_gate_passed > prev:
            for g in range(prev + 1, f.last_gate_passed + 1):
                times[g] = f.t
            prev = f.last_gate_passed
    return times


def completion_percentage(frames: Sequence[TelemetryFrame], total_gates: Optional[int]) -> Optional[float]:
    if not total_gates:
        return None
    passed = gates_passed(frames)
    return 100.0 * min(passed, total_gates) / total_gates


def race_status(frames: Sequence[TelemetryFrame], total_gates: Optional[int]) -> str:
    """Mirrors sim/course.py's print_summary status strings."""
    if not total_gates:
        return "UNKNOWN"
    return "COMPLETE" if gates_passed(frames) >= total_gates else "DNF"


def lap_time(frames: Sequence[TelemetryFrame], total_gates: Optional[int]) -> Optional[float]:
    """Time of the final gate pass if the course was completed, else
    None (mirrors sim/course.py's print_summary: DNF uses final_t
    instead, which callers can get from flight_duration())."""
    if not total_gates or gates_passed(frames) < total_gates:
        return None
    times = gate_pass_times(frames)
    return times.get(total_gates - 1)


def gyro_stability(frames: Sequence[TelemetryFrame]) -> dict:
    """Std-dev and peak of gyro magnitude as a rough proxy for how
    'smooth' vs. 'twitchy' the controller is. Lower std-dev generally
    indicates a better-tuned PID loop, all else equal."""
    mags = [_norm(f.gyro) for f in frames]
    if not mags:
        return {"mean": 0.0, "stdev": 0.0, "peak": 0.0}
    return {"mean": mean(mags), "stdev": pstdev(mags), "peak": max(mags)}


def solver_latency_stats(frames: Sequence[TelemetryFrame]) -> dict:
    """Wall-clock time spent inside autopilot() per tick. Useful for
    catching a perception/planning pipeline that's too slow for
    real-time control, independent of flight quality."""
    lat = [f.solver_latency_s for f in frames if f.solver_latency_s is not None]
    if not lat:
        return {"mean_s": None, "max_s": None, "p95_s": None}
    sorted_lat = sorted(lat)
    p95_idx = min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))
    return {
        "mean_s": mean(lat),
        "max_s": max(lat),
        "p95_s": sorted_lat[p95_idx],
    }


def sensor_freshness_rates(frames: Sequence[TelemetryFrame]) -> dict:
    """Fraction of ticks where each slow sensor actually delivered a
    new reading. Useful for confirming perception/controls are seeing
    the sparse-sensor cadence they expect, e.g. frame_fresh should
    roughly match FPV render rate / physics tick rate."""
    n = len(frames)
    if n == 0:
        return {"gyro": None, "accel": None, "baro": None, "mag": None, "frame": None}
    return {
        "gyro": sum(f.gyro_fresh for f in frames) / n,
        "accel": sum(f.accel_fresh for f in frames) / n,
        "baro": sum(f.baro_fresh for f in frames) / n,
        "mag": sum(f.mag_fresh for f in frames) / n,
        "frame": sum(f.frame_fresh for f in frames) / n,
    }


@dataclass
class RunSummary:
    num_ticks: int
    duration_s: float
    average_speed: float
    max_speed: float
    distance_travelled: float
    gates_passed: int
    total_gates: Optional[int]
    completion_pct: Optional[float]
    race_status: str
    lap_time_s: Optional[float]
    gate_pass_times: dict
    gyro_stability: dict
    solver_latency: dict
    sensor_freshness: dict

    def as_dict(self) -> dict:
        return {
            "num_ticks": self.num_ticks,
            "duration_s": self.duration_s,
            "average_speed": self.average_speed,
            "max_speed": self.max_speed,
            "distance_travelled": self.distance_travelled,
            "gates_passed": self.gates_passed,
            "total_gates": self.total_gates,
            "completion_pct": self.completion_pct,
            "race_status": self.race_status,
            "lap_time_s": self.lap_time_s,
            "gate_pass_times": self.gate_pass_times,
            "gyro_stability": self.gyro_stability,
            "solver_latency": self.solver_latency,
            "sensor_freshness": self.sensor_freshness,
        }


def compute_summary(
    frames: Sequence[TelemetryFrame],
    total_gates: Optional[int] = None,
) -> RunSummary:
    return RunSummary(
        num_ticks=len(frames),
        duration_s=flight_duration(frames),
        average_speed=average_speed(frames),
        max_speed=max_speed(frames),
        distance_travelled=distance_travelled(frames),
        gates_passed=gates_passed(frames),
        total_gates=total_gates,
        completion_pct=completion_percentage(frames, total_gates),
        race_status=race_status(frames, total_gates),
        lap_time_s=lap_time(frames, total_gates),
        gate_pass_times=gate_pass_times(frames),
        gyro_stability=gyro_stability(frames),
        solver_latency=solver_latency_stats(frames),
        sensor_freshness=sensor_freshness_rates(frames),
    )