"""AI Grand Prix Main Autopilot Stack.

Integrates:
  1. Computer Vision (solver.computer_vision)
  2. Path Planning (solver.path_planning)
  3. Flight Control (solver.controls)

Contract:
  autopilot(update: SensorUpdate) -> RCCommand
"""

from __future__ import annotations

from .api import RCCommand, SensorUpdate
from .computer_vision import VisionSystem
from .controls import AutopilotState, FlightController
from .path_planning import PathPlanner

# Phase timing boundaries (seconds since sim start)
T_DISARMED_END = 0.50
T_ARM_IDLE_END = 0.75
T_LAND_END = 14.00

# Global module instances
vision_system = VisionSystem()
path_planner = PathPlanner()
flight_controller = FlightController()


def reset_state() -> None:
    """Reset controller, perception, and planner state."""
    vision_system.reset()
    path_planner.reset()
    flight_controller.reset()


def autopilot(update: SensorUpdate) -> RCCommand:
    """Main tick handler called by simulator every physics iteration."""
    t = update.t

    # 1. State machine management
    if t < T_DISARMED_END:
        state = AutopilotState.DISARMED
    elif t < T_ARM_IDLE_END:
        state = AutopilotState.ARMING_IDLE
    elif t >= T_LAND_END:
        state = AutopilotState.LANDING
    else:
        state = AutopilotState.NAVIGATING

    # Early return for non-flight states
    if state != AutopilotState.NAVIGATING:
        return flight_controller.compute_rc_command(update, target=None, state=state)

    # 2. Computer Vision: Process incoming camera frame if fresh
    detection = None
    if update.frame_fresh and update.frame_rgba is not None:
        detection = vision_system.process_frame(update.frame_rgba)

    # 3. Path Planning: Compute next waypoint target
    target = path_planner.update_trajectory(update, vision_detection=detection)

    # 4. Flight Control: Compute RC channels
    return flight_controller.compute_rc_command(update, target=target, state=state)
