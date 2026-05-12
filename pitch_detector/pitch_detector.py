import os
import pickle
import numpy as np
import supervision as sv
from ultralytics import YOLO


class PitchDetector:
    """
    Detects the soccer field (pitch) keypoints in each video frame using a
    specialized YOLO keypoint detection model.

    This replaces the old ViewTransformer approach that used 4 hard-coded pixel
    coordinates. Instead, the model detects up to 32 keypoints of the field
    (corners, penalty areas, centre circle, etc.) automatically — making the
    system work with any camera angle or field, including Veo recordings.

    The detected keypoints are then used by the ViewTransformer to compute a
    dynamic homography matrix per frame, converting pixel coordinates to real
    world coordinates (centimetres) for accurate speed and distance measurement.

    Args:
        model_path (str): Path to the football-pitch-detection.pt model file.
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> sv.KeyPoints:
        """
        Detects field keypoints in a single frame.

        Args:
            frame (np.ndarray): A single BGR video frame.

        Returns:
            sv.KeyPoints: Detected keypoints. Access pixel coordinates via
                          keypoints.xy[0] (shape: [32, 2]).
                          Points with (0, 0) coordinates were not detected.
        """
        result = self.model(frame, verbose=False)[0]
        return sv.KeyPoints.from_ultralytics(result)

    def get_pitch_keypoints(
        self,
        frames_generator,
        read_from_stub: bool = False,
        stub_path: str = None
    ) -> list:
        """
        Detects field keypoints for each frame, with optional stub caching.

        During development, set read_from_stub=True to avoid re-running the
        model on every run. Set to False (or delete the stub file) to force
        fresh detection.

        Args:
            frames_generator: Iterator yielding BGR video frames.
            read_from_stub (bool): Load from cached pickle file if it exists.
            stub_path (str): Path to the .pkl stub cache file.

        Returns:
            list: List of sv.KeyPoints objects, one per frame.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        all_keypoints = []
        for frame in frames_generator:
            kp = self.detect(frame)
            all_keypoints.append(kp)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(all_keypoints, f)

        return all_keypoints

