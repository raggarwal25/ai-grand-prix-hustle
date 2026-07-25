"""Computer Vision module for AI Grand Prix drone solver.

Processes camera RGBA frames from the forward FPV sensor to detect race gates,
estimate target gate centroids, and project pixel coordinates into 3D body vectors.

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
import numpy as np

# Camera Intrinsics
CAM_WIDTH = 640
CAM_HEIGHT = 360
CAM_FX = 320.0
CAM_FY = 320.0
CAM_CX = 320.0
CAM_CY = 180.0
CAM_TILT_RAD = math.radians(20.0)

# Physical gate dimensions (meters)
GATE_INNER_SIZE = 1.5  # 1.5m x 1.5m inner opening


@dataclass
class VisualGateDetection:
    """Detection result from camera frame analysis."""

    found: bool
    pixel_center: Tuple[float, float] = (320.0, 180.0)
    estimated_distance: float = 0.0
    body_ray: Tuple[float, float, float] = (1.0, 0.0, 0.0)  # Unit vector in body frame [Forward, Left, Up]
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (xmin, ymin, xmax, ymax)
    confidence: float = 0.0


class VisionSystem:
    """Perception stack for processing FPV frames and estimating target gate geometry."""

    def __init__(self) -> None:
        self.last_detection: Optional[VisualGateDetection] = None
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset internal perception state."""
        self.last_detection = None
        self.frame_count = 0

    def pixel_to_body_ray(self, u: float, v: float) -> np.ndarray:
        """Project 2D pixel coordinate (u, v) into a 3D unit ray in body FLU frame.
        
        Body frame conventions (FLU):
          +X = Forward
          +Y = Left
          +Z = Up
        """
        # Pixel relative to principal point (Camera frame: +x right, +y down, +z forward)
        x_cam = (u - CAM_CX) / CAM_FX
        y_cam = (v - CAM_CY) / CAM_FY
        z_cam = 1.0

        # Un-normalized camera ray [Right, Down, Forward] -> [Forward, Left, Up]
        # In level camera frame: Forward = +z_cam, Left = -x_cam, Up = -y_cam
        fwd_cam = z_cam
        left_cam = -x_cam
        up_cam = -y_cam

        # Apply +20° upward tilt pitch rotation around Y_left axis:
        # R_pitch(tilt):
        # fwd_body = fwd_cam * cos(tilt) - up_cam * sin(tilt)
        # up_body  = fwd_cam * sin(tilt) + up_cam * cos(tilt)
        cos_t = math.cos(CAM_TILT_RAD)
        sin_t = math.sin(CAM_TILT_RAD)

        fwd_body = fwd_cam * cos_t - up_cam * sin_t
        left_body = left_cam
        up_body = fwd_cam * sin_t + up_cam * cos_t

        ray = np.array([fwd_body, left_body, up_body], dtype=np.float64)
        norm = np.linalg.norm(ray)
        return ray / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

    def process_frame(self, frame_rgba: Optional[np.ndarray]) -> VisualGateDetection:
        """Analyze RGBA camera frame to locate target race gate.
        
        Uses color thresholding and contour bounding to estimate gate center and distance.
        """
        if frame_rgba is None or frame_rgba.size == 0:
            return VisualGateDetection(found=False)

        self.frame_count += 1

        # Extract RGB components
        r = frame_rgba[:, :, 0].astype(np.int16)
        g = frame_rgba[:, :, 1].astype(np.int16)
        b = frame_rgba[:, :, 2].astype(np.int16)

        # Gate frame mask (e.g. orange / bright red gate frames or high-contrast gate structure)
        # Primary heuristic: High red/orange contrast relative to green/blue
        mask = (r > 120) & (r > g + 20) & (g < 180) & (b < 160)
        
        # Fallback mask if dark/shadowed: high intensity edges or non-zero gate pixels
        mask_count = np.count_nonzero(mask)

        if mask_count < 20:
            # Fallback: check for bright contrasting regions
            intensity = (r + g + b) // 3
            mask = (intensity > 150) & (abs(r - g) > 15)

        y_indices, x_indices = np.where(mask)

        if len(x_indices) < 15:
            detection = VisualGateDetection(found=False)
            self.last_detection = detection
            return detection

        u_center = float(np.mean(x_indices))
        v_center = float(np.mean(y_indices))

        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))

        box_width = max(1, xmax - xmin)
        box_height = max(1, ymax - ymin)

        # Distance estimation using pinhole projection model & known inner gate dimension (1.5m)
        # focal_length * actual_size / pixel_size
        apparent_pixel_size = max(box_width, box_height)
        estimated_dist = (CAM_FX * GATE_INNER_SIZE) / max(1.0, float(apparent_pixel_size))
        estimated_dist = float(np.clip(estimated_dist, 0.5, 50.0))

        body_ray_vec = self.pixel_to_body_ray(u_center, v_center)
        confidence = float(np.clip(mask_count / 500.0, 0.1, 1.0))

        detection = VisualGateDetection(
            found=True,
            pixel_center=(u_center, v_center),
            estimated_distance=estimated_dist,
            body_ray=(float(body_ray_vec[0]), float(body_ray_vec[1]), float(body_ray_vec[2])),
            bounding_box=(xmin, ymin, xmax, ymax),
            confidence=confidence,
        )

        self.last_detection = detection
        return detection
