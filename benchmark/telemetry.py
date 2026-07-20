"""
telemetry.py
------------
Defines the per-tick telemetry record (`TelemetryFrame`) and the
`TelemetryCollector` that captures one frame every physics tick.

Field names here mirror `solver/api.py`'s `SensorUpdate`/`RCCommand`
exactly:

    SensorUpdate.t, .tick
    SensorUpdate.world_pos   -> np.ndarray[7]: [qx, qy, qz, qw, x, y, z]
    SensorUpdate.world_vel   -> np.ndarray[6]: [wx, wy, wz, vx, vy, vz]
    SensorUpdate.gyro, .accel -> np.ndarray[3]
    SensorUpdate.baro         -> float
    SensorUpdate.mag          -> np.ndarray[3]
    SensorUpdate.{gyro,accel,baro,mag,frame}_fresh -> bool
    SensorUpdate.last_gate_passed, .next_gate_index -> int

    RCCommand.throttle/roll/pitch/yaw/arm/aux2/aux3/aux4 -> int (PWM us)

This module still duck-types via getattr rather than importing
`solver.api` directly, so a future contract change doesn't hard-crash
the benchmark tool — it'll just fall back to zeros for anything
renamed, which is visible in the exported CSV rather than silent.

Note: there is currently no collision/crash signal exposed anywhere
in this sim. We do not fabricate one — `collided` stays None until
the sim exposes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import time

import numpy as np

Vec3 = Tuple[float, float, float]


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Defensive attribute access — see module docstring."""
    return getattr(obj, name, default)


def _as_array(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size < length:
        arr = np.pad(arr, (0, length - arr.size))
    return arr[:length]


def position_from_world_pos(world_pos: np.ndarray) -> Vec3:
    """world_pos layout is [qx, qy, qz, qw, x, y, z] (Elodin scalar-last
    quat + ENU position)."""
    wp = _as_array(world_pos, 7)
    return (float(wp[4]), float(wp[5]), float(wp[6]))


def orientation_from_world_pos(world_pos: np.ndarray) -> Tuple[float, float, float, float]:
    wp = _as_array(world_pos, 7)
    return (float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3]))


def angular_velocity_from_world_vel(world_vel: np.ndarray) -> Vec3:
    wv = _as_array(world_vel, 6)
    return (float(wv[0]), float(wv[1]), float(wv[2]))


def linear_velocity_from_world_vel(world_vel: np.ndarray) -> Vec3:
    wv = _as_array(world_vel, 6)
    return (float(wv[3]), float(wv[4]), float(wv[5]))


@dataclass(slots=True)
class TelemetryFrame:
    """One row of telemetry: everything we know at a single physics tick."""

    tick: int
    t: float
    wall_time: float  # time.perf_counter() at capture, for real-time profiling

    position: Vec3            # derived from world_pos[4:7]
    linear_velocity: Vec3      # derived from world_vel[3:6]
    angular_velocity: Vec3     # derived from world_vel[0:3]
    gyro: Vec3
    accel: Vec3
    baro: float
    mag: Vec3

    gyro_fresh: bool = True
    accel_fresh: bool = True
    baro_fresh: bool = False
    mag_fresh: bool = False
    frame_fresh: bool = False

    last_gate_passed: int = -1
    next_gate_index: int = -1

    # RC output actually issued this tick (post-solver).
    rc_throttle: Optional[int] = None
    rc_roll: Optional[int] = None
    rc_pitch: Optional[int] = None
    rc_yaw: Optional[int] = None
    rc_arm: Optional[int] = None

    solver_latency_s: Optional[float] = None

    # No collision signal exists in this sim version yet.
    collided: Optional[bool] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "tick": self.tick,
            "t": self.t,
            "wall_time": self.wall_time,
            "pos_x": self.position[0],
            "pos_y": self.position[1],
            "pos_z": self.position[2],
            "vel_x": self.linear_velocity[0],
            "vel_y": self.linear_velocity[1],
            "vel_z": self.linear_velocity[2],
            "ang_vel_x": self.angular_velocity[0],
            "ang_vel_y": self.angular_velocity[1],
            "ang_vel_z": self.angular_velocity[2],
            "gyro_x": self.gyro[0],
            "gyro_y": self.gyro[1],
            "gyro_z": self.gyro[2],
            "accel_x": self.accel[0],
            "accel_y": self.accel[1],
            "accel_z": self.accel[2],
            "baro": self.baro,
            "mag_x": self.mag[0],
            "mag_y": self.mag[1],
            "mag_z": self.mag[2],
            "gyro_fresh": self.gyro_fresh,
            "accel_fresh": self.accel_fresh,
            "baro_fresh": self.baro_fresh,
            "mag_fresh": self.mag_fresh,
            "frame_fresh": self.frame_fresh,
            "last_gate_passed": self.last_gate_passed,
            "next_gate_index": self.next_gate_index,
            "rc_throttle": self.rc_throttle,
            "rc_roll": self.rc_roll,
            "rc_pitch": self.rc_pitch,
            "rc_yaw": self.rc_yaw,
            "rc_arm": self.rc_arm,
            "solver_latency_s": self.solver_latency_s,
            "collided": self.collided,
        }
        d.update(self.extra)
        return d


class TelemetryCollector:
    """Accumulates TelemetryFrames for a single run.

    Usage:
        collector = TelemetryCollector()
        collector.collect(sensor_update, rc_out=rc_command, latency_s=dt)
        ...
        frames = collector.frames
    """

    def __init__(self) -> None:
        self._frames: list[TelemetryFrame] = []

    def collect(
        self,
        update: Any,
        rc_out: Optional[Any] = None,
        latency_s: Optional[float] = None,
    ) -> TelemetryFrame:
        world_pos = _get(update, "world_pos")
        world_vel = _get(update, "world_vel")

        rc_throttle = _get(rc_out, "throttle")
        rc_roll = _get(rc_out, "roll")
        rc_pitch = _get(rc_out, "pitch")
        rc_yaw = _get(rc_out, "yaw")
        rc_arm = _get(rc_out, "arm")

        frame = TelemetryFrame(
            tick=int(_get(update, "tick", len(self._frames))),
            t=float(_get(update, "t", 0.0)),
            wall_time=time.perf_counter(),
            position=position_from_world_pos(world_pos),
            linear_velocity=linear_velocity_from_world_vel(world_vel),
            angular_velocity=angular_velocity_from_world_vel(world_vel),
            gyro=tuple(_as_array(_get(update, "gyro"), 3).tolist()),
            accel=tuple(_as_array(_get(update, "accel"), 3).tolist()),
            baro=float(_get(update, "baro", 0.0)),
            mag=tuple(_as_array(_get(update, "mag"), 3).tolist()),
            gyro_fresh=bool(_get(update, "gyro_fresh", True)),
            accel_fresh=bool(_get(update, "accel_fresh", True)),
            baro_fresh=bool(_get(update, "baro_fresh", False)),
            mag_fresh=bool(_get(update, "mag_fresh", False)),
            frame_fresh=bool(_get(update, "frame_fresh", False)),
            last_gate_passed=int(_get(update, "last_gate_passed", -1)),
            next_gate_index=int(_get(update, "next_gate_index", -1)),
            rc_throttle=rc_throttle,
            rc_roll=rc_roll,
            rc_pitch=rc_pitch,
            rc_yaw=rc_yaw,
            rc_arm=rc_arm,
            solver_latency_s=latency_s,
        )
        self._frames.append(frame)
        return frame

    @property
    def frames(self) -> list[TelemetryFrame]:
        return self._frames

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()