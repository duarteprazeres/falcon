"""
Ball tracker with Kalman filter prediction.

Fills gaps of 2–MAX_GAP frames where the ball is not detected by predicting
its position based on the previous velocity, corrected for camera movement.

State vector: [x, y, vx, vy]  (centre pixel, pixels/frame)
Measurement:  [x, y]
"""

import numpy as np
from filterpy.kalman import KalmanFilter

MAX_GAP = 10  # predict for at most this many consecutive missing frames (~0.33 s at 30fps)


class BallKalmanTracker:
    """
    Wraps a simple constant-velocity Kalman filter for the ball.

    Usage (called once per frame in sequence):
        tracker = BallKalmanTracker()
        for frame_num, frame in enumerate(frames):
            detected_bbox = ...  # [x1,y1,x2,y2] or None
            cam_dx, cam_dy = camera_movement_per_frame[frame_num]
            cx, cy = tracker.update(detected_bbox, cam_dx, cam_dy)
            # cx, cy  →  best estimate of ball centre (detected or predicted)
            # None    →  gap too long, no reliable prediction
    """

    def __init__(self):
        # State: [x, y, vx, vy]
        kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0   # one frame time-step

        # State-transition matrix  (constant velocity)
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0,  1, 0],
            [0, 0,  0, 1],
        ], dtype=np.float64)

        # Measurement matrix  (we observe x, y directly)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Measurement noise  (YOLO localisation uncertainty ~5 px)
        kf.R = np.eye(2) * 25.0

        # Process noise  (ball can accelerate; tune Q for agility vs. smoothness)
        kf.Q = np.diag([1.0, 1.0, 10.0, 10.0])

        # Initial covariance (high uncertainty before first detection)
        kf.P = np.eye(4) * 500.0

        self._kf = kf
        self._initialised = False
        self._frames_since_detection = 0

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _centre(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    # ------------------------------------------------------------------ public

    def update(self, bbox, cam_dx: float, cam_dy: float):
        """
        Call once per frame.

        bbox    : [x1, y1, x2, y2] if ball detected, else None.
        cam_dx  : camera pan in x (pixels) this frame (from camera_movement_estimator).
        cam_dy  : camera pan in y (pixels) this frame.

        Returns (cx, cy) — the best estimate of the ball centre — or None if
        there have been more than MAX_GAP consecutive missing detections.
        """
        # --- Compensate for camera movement in the Kalman state ---
        # A camera pan of (dx, dy) shifts ALL pixel positions by (dx, dy).
        # The ball hasn't physically moved, but its pixel position has.
        # We update the prior state with this shift before predicting.
        if self._initialised and (cam_dx != 0.0 or cam_dy != 0.0):
            self._kf.x[0] += cam_dx
            self._kf.x[1] += cam_dy

        # --- Kalman predict step ---
        if self._initialised:
            self._kf.predict()

        if bbox is not None:
            cx, cy = self._centre(bbox)
            z = np.array([[cx], [cy]], dtype=np.float64)

            if not self._initialised:
                # First detection — initialise state
                self._kf.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
                self._kf.P = np.eye(4) * 500.0
                self._initialised = True
            else:
                self._kf.update(z)

            self._frames_since_detection = 0
            return float(self._kf.x[0, 0]), float(self._kf.x[1, 0])

        else:
            # No detection
            if not self._initialised:
                return None

            self._frames_since_detection += 1
            if self._frames_since_detection > MAX_GAP:
                return None  # prediction no longer reliable

            # Return predicted position (already computed in predict step above)
            return float(self._kf.x[0, 0]), float(self._kf.x[1, 0])
