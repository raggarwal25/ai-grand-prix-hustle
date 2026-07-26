"""Computer Vision module for AI Grand Prix drone solver.

Processes camera RGBA frames from the forward FPV sensor, estimates 6D gate pose using
OpenCV Perspective-n-Point (solvePnP), and filters 3D target states using a Kalman Filter.

Camera Intrinsics (VADR-TS-002 §3.8):
  - Resolution: 640 x 360
  - Principal point (cx, cy): 320.0, 180.0
  - Focal lengths (fx, fy): 320.0, 320.0
  - Tilt: +20 degrees upward in body frame
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

# Camera Intrinsics
CAM_WIDTH = 640
CAM_HEIGHT = 360
CAM_FX = 320.0
CAM_FY = 320.0
CAM_CX = 320.0
CAM_CY = 180.0
CAM_TILT_RAD = math.radians(20.0)

# Camera intrinsic matrix K
CAMERA_MATRIX = np.array(
    [
        [CAM_FX, 0.0, CAM_CX],
        [0.0, CAM_FY, CAM_CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# Physical inner gate dimensions (1.5m x 1.5m)
GATE_INNER_SIZE = 1.5
HALF_W = GATE_INNER_SIZE / 2.0
HALF_H = GATE_INNER_SIZE / 2.0

# 3D Model Points of gate inner opening corners (in gate-local frame)
GATE_3D_CORNERS = np.array(
    [
        [-HALF_W, +HALF_H, 0.0],
        [+HALF_W, +HALF_H, 0.0],
        [+HALF_W, -HALF_H, 0.0],
        [-HALF_W, -HALF_H, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class VisualGateDetection:
    """Detection result from camera frame analysis."""

    found: bool
    pixel_center: Tuple[float, float] = (320.0, 180.0)
    estimated_distance: float = 0.0
    body_ray: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Unit vector in body frame [Forward, Left, Up]
    body_pos_pnp: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D relative position [Forward, Left, Up]
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (xmin, ymin, xmax, ymax)
    confidence: float = 0.0


class KalmanGateFilter:
    """6-DOF Kalman Filter for tracking 3D relative target gate position and velocity."""

    def __init__(self) -> None:
        self.state = np.zeros(6, dtype=np.float64)  # [x, y, z, vx, vy, vz]
        self.P = np.eye(6, dtype=np.float64) * 1.0  # Covariance matrix
        self.Q = np.eye(6, dtype=np.float64) * 0.05  # Process noise
        self.R = np.eye(3, dtype=np.float64) * 0.1  # Measurement noise
        self.H = np.zeros((3, 6), dtype=np.float64)  # Measurement matrix
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.initialized = False

    def reset(self) -> None:
        """Reset state and covariance."""
        self.state.fill(0.0)
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = False

    def predict(self, dt: float = 0.033) -> np.ndarray:
        """Predict state forward in time."""
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q
        return self.state[:3]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update state estimate with 3D measurement [x, y, z]."""
        if not self.initialized:
            self.state[:3] = measurement
            self.state[3:] = 0.0
            self.initialized = True
            return self.state[:3]

        y = measurement - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return self.state[:3]


class VisionSystem:
    """Perception stack for processing FPV frames, PnP 6D pose estimation, and target filtering."""

    def __init__(self) -> None:
        self.last_detection: Optional[VisualGateDetection] = None
        self.kalman_filter = KalmanGateFilter()
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset internal perception state."""
        self.last_detection = None
        self.kalman_filter.reset()
        self.frame_count = 0

    def pixel_to_body_ray(self, u: float, v: float) -> np.ndarray:
        """Project 2D pixel coordinate (u, v) into a 3D unit ray in body FLU frame.
        
        Body frame conventions (FLU):
          +X = Forward, +Y = Left, +Z = Up
        """
        x_cam = (u - CAM_CX) / CAM_FX
        y_cam = (v - CAM_CY) / CAM_FY
        z_cam = 1.0

        fwd_cam = z_cam
        left_cam = -x_cam
        up_cam = -y_cam

        cos_t = math.cos(CAM_TILT_RAD)
        sin_t = math.sin(CAM_TILT_RAD)

        fwd_body = fwd_cam * cos_t - up_cam * sin_t
        left_body = left_cam
        up_body = fwd_cam * sin_t + up_cam * cos_t

        ray = np.array([fwd_body, left_body, up_body], dtype=np.float64)
        norm = np.linalg.norm(ray)
        return ray / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    def camera_to_body_pos(self, tvec: np.ndarray) -> np.ndarray:
        """Convert camera-frame translation vector [X_right, Y_down, Z_fwd] to body FLU [X_fwd, Y_left, Z_up]."""
        x_cam, y_cam, z_cam = tvec.flatten()

        fwd_cam = z_cam
        left_cam = -x_cam
        up_cam = -y_cam

        cos_t = math.cos(CAM_TILT_RAD)
        sin_t = math.sin(CAM_TILT_RAD)

        fwd_body = fwd_cam * cos_t - up_cam * sin_t
        left_body = left_cam
        up_body = fwd_cam * sin_t + up_cam * cos_t

        return np.array([fwd_body, left_body, up_body], dtype=np.float64)

    def process_frame(self, frame_rgba: Optional[np.ndarray]) -> VisualGateDetection:
        """Analyze RGBA camera frame using OpenCV solvePnP to estimate exact 6D gate pose."""
        if frame_rgba is None or frame_rgba.size == 0:
            return VisualGateDetection(found=False)

        self.frame_count += 1

        r = frame_rgba[:, :, 0]
        g = frame_rgba[:, :, 1]
        b = frame_rgba[:, :, 2]

        mask = (r > 120) & (r > g) & (g < 180) & (b < 160)
        mask_count = np.count_nonzero(mask)

        if mask_count < 20:
            mask = (r > 150) & (g > 150)
            mask_count = np.count_nonzero(mask)

        y_indices, x_indices = np.where(mask)

        if len(x_indices) < 15:
            detection = VisualGateDetection(found=False)
            self.last_detection = detection
            return detection

        u_center = float(np.mean(x_indices))
        v_center = float(np.mean(y_indices))

        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))

        # 2D Bounding Box corners in image space
        img_corners = np.array(
            [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ],
            dtype=np.float64,
        )

        # OpenCV solvePnP 6D pose estimation
        success, rvec, tvec = cv2.solvePnP(
            GATE_3D_CORNERS,
            img_corners,
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success and tvec is not None:
            raw_body_pos = self.camera_to_body_pos(tvec)
            # Filter 3D measurement with Kalman Filter
            filtered_body_pos = self.kalman_filter.update(raw_body_pos)
            estimated_dist = float(np.linalg.norm(filtered_body_pos))
        else:
            box_w = max(1, xmax - xmin)
            estimated_dist = float(np.clip((CAM_FX * GATE_INNER_SIZE) / float(box_w), 0.5, 50.0))
            filtered_body_pos = self.pixel_to_body_ray(u_center, v_center) * estimated_dist

        body_ray_vec = self.pixel_to_body_ray(u_center, v_center)
        confidence = float(np.clip(mask_count / 500.0, 0.1, 1.0))

        detection = VisualGateDetection(
            found=True,
            pixel_center=(u_center, v_center),
            estimated_distance=estimated_dist,
            body_ray=(float(body_ray_vec[0]), float(body_ray_vec[1]), float(body_ray_vec[2])),
            body_pos_pnp=(float(filtered_body_pos[0]), float(filtered_body_pos[1]), float(filtered_body_pos[2])),
            bounding_box=(xmin, ymin, xmax, ymax),
            confidence=confidence,
        )

        self.last_detection = detection
        return detection
