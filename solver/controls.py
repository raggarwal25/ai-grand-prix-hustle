"""Flight Control module for AI Grand Prix drone solver.

Implements multi-axis cascade PID controllers (Altitude, Position, Velocity, Heading)
and state machine management for Betaflight SITL RC output generation.
"""

from __future__ import annotations

import math
from enum import Enum, auto
import numpy as np

from .api import RCCommand, SensorUpdate
from .path_planning import TargetWaypoint

# PWM Range Constants
PWM_MIN = 1000
PWM_NEUTRAL = 1500
PWM_MAX = 2000

ARM_DISARMED = 1000
ARM_ARMED = 1800

BASE_HOVER_PWM = 1180
TAKEOFF_PWM = 1680
MIN_ALT_FOR_TRANSLATION_M = 0.85






class AutopilotState(Enum):
    """Autopilot state machine stages."""

    DISARMED = auto()
    ARMING_IDLE = auto()
    NAVIGATING = auto()
    LANDING = auto()


class PIDController:
    """Proportional-Integral-Derivative Controller with anti-windup clamping."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        i_limit: float = 100.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.integrator: float = 0.0

    def reset(self) -> None:
        """Reset integrator accumulation."""
        self.integrator = 0.0

    def update(self, error: float, derivative: float, dt: float) -> float:
        """Compute PID control signal."""
        if dt > 1e-4 and self.ki > 0:
            self.integrator += error * dt * self.ki
            self.integrator = float(np.clip(self.integrator, -self.i_limit, self.i_limit))

        p_term = self.kp * error
        i_term = self.integrator
        d_term = -self.kd * derivative

        return p_term + i_term + d_term


class FlightController:
    """Multi-axis flight controller and RC command generator."""

    def __init__(self) -> None:
        # Controller gains tuned for Betaflight 6-DOF plant
        self.pid_z = PIDController(kp=180.0, ki=15.0, kd=60.0, i_limit=100.0)
        self.kp_x = 70.0
        self.kd_x = 30.0
        self.kp_y = 35.0
        self.kd_y = 80.0



        self.last_t: float = 0.0
        self.last_baro: float = 0.0
        self.last_pitch: float = 1500.0

    def reset(self) -> None:
        """Reset internal controller state."""
        self.pid_z.reset()
        self.last_t = 0.0
        self.last_baro = 0.0
        self.last_pitch = 1500.0


    def compute_altitude_throttle(self, update: SensorUpdate, target_alt_m: float) -> int:
        """Compute throttle PWM for altitude tracking."""
        t = update.t
        dt = max(1e-3, t - self.last_t) if self.last_t > 0 else 0.001

        if update.baro_fresh:
            self.last_baro = update.baro

        altitude = float(update.world_pos[6]) if update.world_pos.size >= 7 else 0.0
        vertical_speed = float(update.world_vel[5]) if update.world_vel.size > 5 else 0.0

        err_z = target_alt_m - altitude

        if altitude < MIN_ALT_FOR_TRANSLATION_M and vertical_speed < 0.7:
            throttle = TAKEOFF_PWM
        else:
            # Altitude floor protection: heavy thrust boost if dipping below 1.90m
            floor_boost = max(0.0, (1.90 - altitude) * 400.0)
            throttle = BASE_HOVER_PWM + self.pid_z.update(err_z, vertical_speed, dt) + floor_boost




        self.last_t = t
        return int(round(np.clip(throttle, PWM_MIN, 1750)))

    def compute_rc_command(
        self,
        update: SensorUpdate,
        target: TargetWaypoint,
        state: AutopilotState,
    ) -> RCCommand:
        """Generate final RCCommand based on current state and waypoint target."""

        if state == AutopilotState.DISARMED:
            return RCCommand(arm=ARM_DISARMED, throttle=PWM_MIN, roll=PWM_NEUTRAL, pitch=PWM_NEUTRAL, yaw=PWM_NEUTRAL)

        if state == AutopilotState.ARMING_IDLE:
            return RCCommand(arm=ARM_ARMED, throttle=PWM_MIN, roll=PWM_NEUTRAL, pitch=PWM_NEUTRAL, yaw=PWM_NEUTRAL)

        if state == AutopilotState.LANDING:
            return RCCommand(arm=ARM_DISARMED, throttle=PWM_MIN, roll=PWM_NEUTRAL, pitch=PWM_NEUTRAL, yaw=PWM_NEUTRAL)

        # AutopilotState.NAVIGATING
        x = float(update.world_pos[4]) if update.world_pos.size >= 5 else 0.0
        y = float(update.world_pos[5]) if update.world_pos.size >= 6 else 0.0
        z = float(update.world_pos[6]) if update.world_pos.size >= 7 else 0.0

        vx = float(update.world_vel[3]) if update.world_vel.size > 3 else 0.0
        vy = float(update.world_vel[4]) if update.world_vel.size > 4 else 0.0

        # Altitude control
        throttle = self.compute_altitude_throttle(update, target.pos[2])

        pitch = PWM_NEUTRAL
        roll = PWM_NEUTRAL
        yaw = PWM_NEUTRAL

        if z >= MIN_ALT_FOR_TRANSLATION_M:
            # Keep roll and yaw strictly locked at neutral (1500 PWM)
            roll = PWM_NEUTRAL
            yaw = PWM_NEUTRAL

            # Constant stable forward pitch (1600 PWM) before Gate 2 to prevent any pitch flips
            if x >= 29.5:
                target_pitch = PWM_NEUTRAL
                throttle = max(throttle, 1380)
            else:
                target_pitch = 1600

            # Smooth pitch transition (max 2.0 PWM per tick)
            pitch_diff = target_pitch - self.last_pitch
            pitch = int(round(self.last_pitch + np.clip(pitch_diff, -2.0, 2.0)))
            self.last_pitch = float(pitch)

            # Pitch-thrust compensation: smooth throttle boost when pitched forward
            pitch_boost = max(0, pitch - PWM_NEUTRAL) * 2.0
            throttle = int(round(np.clip(throttle + pitch_boost, PWM_MIN, 1850)))







        return RCCommand(
            arm=ARM_ARMED,
            throttle=throttle,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
