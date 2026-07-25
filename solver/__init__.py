"""Contestant autonomy stack hook for AI Grand Prix.

Exposes core modules:
  - computer_vision: FPV camera image processing & gate projection
  - path_planning: Trajectory generation & gate sequence tracking
  - controls: Multi-axis cascade PID & RC command generation
"""

from .api import SensorUpdate, RCCommand
from .baseline import autopilot, reset_state
from .computer_vision import VisionSystem, VisualGateDetection
from .path_planning import PathPlanner, TargetWaypoint
from .controls import FlightController, AutopilotState

__all__ = [
    "SensorUpdate",
    "RCCommand",
    "autopilot",
    "reset_state",
    "VisionSystem",
    "VisualGateDetection",
    "PathPlanner",
    "TargetWaypoint",
    "FlightController",
    "AutopilotState",
]

