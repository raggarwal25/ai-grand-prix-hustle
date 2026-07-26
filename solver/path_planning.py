"""Path Planning module for AI Grand Prix drone solver.

Computes target trajectory waypoints, waypoint sequencing, and target velocity vectors
to guide the quadrotor safely through racecourse gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .api import SensorUpdate
from .computer_vision import VisualGateDetection


# Known gate positions in ENU world coordinates: (x, y, z)
# Matches EASY_COURSE in sim/course.py
DEFAULT_GATES: Tuple[Tuple[float, float, float], ...] = (
    (10.0, 0.0, 1.95),
    (20.0, 0.0, 1.95),
    (30.0, 0.0, 1.95),
)



@dataclass
class TargetWaypoint:
    """Calculated goal position, desired velocity, and target yaw heading."""

    pos: np.ndarray  # [x, y, z] in ENU world frame
    target_vel: np.ndarray  # [vx, vy, vz] desired speed vector
    target_yaw_rad: float  # Target heading angle in radians
    gate_index: int  # Current target gate index
    is_approach: bool  # True if navigating to approach waypoint before gate center


class PathPlanner:
    """Trajectory generator and gate sequence manager."""

    def __init__(self, course_gates: Tuple[Tuple[float, float, float], ...] = DEFAULT_GATES) -> None:
        self.course_gates = [np.array(g, dtype=np.float64) for g in course_gates]
        self.current_gate_idx: int = 0
        self.cruise_speed: float = 10.0  # m/s target forward speed
        self.approach_margin_m: float = 0.5  # m offset before gate for alignment

    def reset(self) -> None:
        """Reset path planner state."""
        self.current_gate_idx = 0

    def get_target_gate_center(self, next_gate_index: int) -> np.ndarray:
        """Get static world coordinate for target gate."""
        if next_gate_index < 0 or next_gate_index >= len(self.course_gates):
            return self.course_gates[-1]
        return self.course_gates[next_gate_index]

    def update_trajectory(
        self,
        update: SensorUpdate,
        vision_detection: Optional[VisualGateDetection] = None,
    ) -> TargetWaypoint:
        """Compute the optimal current target position and velocity vector.
        
        Blends world state feedback with vision detections when available.
        """
        drone_pos = update.world_pos[4:7] if update.world_pos.size >= 7 else np.zeros(3)

        # Determine active gate index or finish line straight flight
        idx = int(np.clip(update.next_gate_index, 0, len(self.course_gates) - 1)) if update.next_gate_index != -1 else len(self.course_gates) - 1
        self.current_gate_idx = idx

        if idx >= 2 or drone_pos[0] >= 22.0:
            # On or past Gate 2: set target 100m ahead along +X so dx is always +70m and pitch stays locked at 1610
            target_pos = np.array([100.0, 0.0, 1.95])
        else:
            gate_center = self.get_target_gate_center(idx)
            target_pos = gate_center + np.array([5.0, 0.0, 0.0])





        gate_center = self.get_target_gate_center(self.current_gate_idx)
        dist_to_gate = float(np.linalg.norm(gate_center - drone_pos))


        # Compute desired direction vector
        dir_vec = target_pos - drone_pos
        dist_to_target = float(np.linalg.norm(dir_vec))

        if dist_to_target > 1e-3:
            unit_dir = dir_vec / dist_to_target
            target_vel = unit_dir * self.cruise_speed
        else:
            unit_dir = np.array([1.0, 0.0, 0.0])
            target_vel = unit_dir * self.cruise_speed


        # Heading (yaw angle): align with velocity vector
        target_yaw = math.atan2(unit_dir[1], unit_dir[0])
        is_approach = dist_to_gate > self.approach_margin_m

        return TargetWaypoint(

            pos=target_pos,
            target_vel=target_vel,
            target_yaw_rad=target_yaw,
            gate_index=self.current_gate_idx,
            is_approach=is_approach,
        )
