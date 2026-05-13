import json
import os
import numpy as np
import cv2
import supervision as sv
from sports.common.view import ViewTransformer as _RoboflowViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()

CALIBRATION_PATH = "calibration/calibration.json"


class ViewTransformer:
    """
    Dynamic perspective transformer: pixel coordinates → real-world cm.

    Supports two modes:
    1. YOLO mode (default): homography recomputed each frame from detected keypoints.
    2. Calibration mode: homography computed once from manual calibration, then
       adjusted per frame using cumulative camera movement (optical flow).
       Skips the YOLO pitch detection pass entirely — ~40% faster.
    """

    def __init__(self):
        self.config = CONFIG
        self._transformer = None

        # Calibration mode state
        self._cal_source_ref = None   # reference pixel points (N×2 float32)
        self._cal_target = None       # world points (N×2 float32)
        self._calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def load_calibration(self, path: str = CALIBRATION_PATH) -> bool:
        """
        Loads a manual calibration saved by calibrate.py.

        Returns True if loaded successfully, False if the file doesn't exist.
        """
        if not os.path.exists(path):
            return False

        with open(path) as f:
            data = json.load(f)

        self._cal_source_ref = np.array(data["pixel_points"], dtype=np.float32)
        self._cal_target = np.array(data["world_points"], dtype=np.float32)
        self._calibrated = True

        # Build the initial homography from the reference frame
        self._build_calibrated_homography(np.zeros(2, dtype=np.float32))
        print(f"Calibração carregada: {data['n_points']} pontos de '{path}'")
        return True

    def _build_calibrated_homography(self, cumulative_movement: np.ndarray) -> bool:
        """
        Recomputes the homography from the calibration points adjusted by the
        cumulative camera panning offset since the reference frame.

        For a panning camera, a displacement of (dx, dy) pixels means every
        reference pixel point needs to be shifted by (-dx, -dy) to find where
        that same real-world point appears in the current frame.
        """
        shifted_source = self._cal_source_ref - cumulative_movement
        try:
            self._transformer = _RoboflowViewTransformer(
                source=shifted_source,
                target=self._cal_target,
            )
            return True
        except (ValueError, cv2.error):
            return False

    def get_calibration_pixel_hull(self) -> np.ndarray | None:
        """
        Returns the convex hull of the calibration source points in pixel space.

        Used to filter out-of-field detections when in calibration mode (where
        YOLO pitch keypoints are not available per-frame).
        """
        if not self._calibrated or self._cal_source_ref is None or len(self._cal_source_ref) < 3:
            return None
        return cv2.convexHull(self._cal_source_ref.astype(np.float32))

    def update_from_camera_movement(self, cumulative_dx: float, cumulative_dy: float) -> bool:
        """
        Updates the homography for the current frame using the total camera
        displacement (in pixels) accumulated since the reference frame.

        Call this once per frame instead of update() when in calibration mode.
        """
        if not self._calibrated:
            return False
        movement = np.array([cumulative_dx, cumulative_dy], dtype=np.float32)
        return self._build_calibrated_homography(movement)

    def update(self, keypoints: sv.KeyPoints) -> bool:
        """
        Updates the homography matrix using the field keypoints detected in
        the current frame.

        Only keypoints with a confidence above zero (i.e., actually detected)
        are used. Homography requires at least 4 matched points.

        Args:
            keypoints (sv.KeyPoints): Field keypoints from PitchDetector.detect().

        Returns:
            bool: True if homography was successfully computed, False otherwise
                  (e.g., fewer than 4 keypoints detected in this frame).
        """
        if len(keypoints.xy) == 0:
            return False

        # Filter out undetected keypoints (those at pixel 0,0 or with zero confidence)
        xy = keypoints.xy[0]  # shape [N, 2]
        mask = (xy[:, 0] > 1) & (xy[:, 1] > 1)

        source_pts = xy[mask].astype(np.float32)
        target_pts = np.array(self.config.vertices)[mask].astype(np.float32)

        if len(source_pts) < 4:
            # Not enough keypoints visible to compute homography — camera may be
            # too zoomed in or the field is partially out of frame.
            return False

        try:
            self._transformer = _RoboflowViewTransformer(
                source=source_pts,
                target=target_pts
            )
            return True
        except ValueError:
            return False

    def transform_point(self, point: np.ndarray):
        """
        Transforms a single pixel-space point to real-world field coordinates.

        Args:
            point (np.ndarray): A [x, y] pixel coordinate.

        Returns:
            np.ndarray or None: Real-world [x, y] in cm, or None if no valid
                                homography is available for this frame.
        """
        if self._transformer is None:
            return None

        pts = np.array([[point[0], point[1]]], dtype=np.float32)
        transformed = self._transformer.transform_points(pts)
        return transformed[0]

    def add_transformed_position_to_tracks(self, tracks: dict) -> None:
        """
        Adds 'position_transformed' (real-world cm coordinates) to each
        tracked object using the most recently updated homography.

        This is called once per frame after update() has been called with
        the current frame's keypoints.

        Args:
            tracks (dict): The tracking dictionary with structure:
                           {object_type: [{track_id: {bbox, position_adjusted, ...}}]}
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info.get('position_adjusted')
                    if position is None:
                        tracks[object_type][frame_num][track_id]['position_transformed'] = None
                        continue

                    position_np = np.array(position, dtype=np.float32)
                    transformed = self.transform_point(position_np)

                    if transformed is not None:
                        transformed = transformed.tolist()
                    tracks[object_type][frame_num][track_id]['position_transformed'] = transformed