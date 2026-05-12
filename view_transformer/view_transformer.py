import numpy as np
import cv2
import supervision as sv
from sports.common.view import ViewTransformer as _RoboflowViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# The standard soccer pitch configuration with real-world dimensions in cm.
# A full FIFA pitch is 120m × 70m (12000cm × 7000cm).
# You can override these dimensions for smaller youth pitches if needed.
CONFIG = SoccerPitchConfiguration()


class ViewTransformer:
    """
    Dynamic perspective transformer that converts pixel coordinates in the
    video frame to real-world field coordinates (centimetres).

    Unlike the previous implementation (which had 4 hard-coded pixel vertices
    specific to a single video), this version:
    - Receives detected field keypoints from PitchDetector on each frame
    - Automatically computes the homography matrix from those keypoints
    - Works with ANY camera angle, zoom level, or field (including Veo)

    The real-world coordinate system is defined by SoccerPitchConfiguration:
    - Origin (0, 0): top-left corner of the pitch
    - X axis: along the length of the pitch (0 → 12000 cm)
    - Y axis: along the width of the pitch (0 → 7000 cm)
    """

    def __init__(self):
        self.config = CONFIG
        self._transformer = None  # Built dynamically per frame

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