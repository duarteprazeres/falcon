import pickle
import cv2
import numpy as np
import os

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 1.0

        self.lk_params = dict(
            winSize = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = first_frame_grayscale.shape
        mask_features = np.zeros_like(first_frame_grayscale)

        # Track static background features (sky, stands, advertising boards)
        # Enlarged to 15% so more stable background area is sampled.
        mask_features[0:h//7, :] = 1           # Top 15%
        mask_features[h - h//7:, :] = 1        # Bottom 15%
        mask_features[:, 0:w//7] = 1           # Left 15%
        mask_features[:, w - w//7:] = 1        # Right 15%

        self.features = dict(
            maxCorners = 200,
            qualityLevel = 0.01,
            minDistance = 5,
            blockSize = 7,
            mask = mask_features
        )

        # Stateful fields for process_frame() (single-pass mode)
        self._old_gray = first_frame_grayscale
        self._old_features = cv2.goodFeaturesToTrack(first_frame_grayscale, **self.features)

    def _estimate_movement(self, old_gray, frame_gray, old_features):
        """
        Robustly estimates camera displacement between two frames using
        the MEDIAN of all successfully tracked feature displacements.

        Using the median (instead of the max) is critical: the maximum
        displacement picks the fastest-moving object in the border region
        (could be a player or referee), while the median reflects the
        dominant global shift shared by all static background features.
        """
        if old_features is None or len(old_features) == 0:
            return [0.0, 0.0], None

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, old_features, None, **self.lk_params
        )

        if new_features is None or status is None:
            return [0.0, 0.0], None

        # Keep only features that were successfully tracked
        good_new = new_features[status.ravel() == 1]
        good_old = old_features[status.ravel() == 1]

        if len(good_new) < 4:
            return [0.0, 0.0], None

        movements = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
        max_distance = float(np.max(np.linalg.norm(movements, axis=1)))

        if max_distance < self.minimum_distance:
            return [0.0, 0.0], good_new

        # Median is robust to outlier features (e.g., a player near the border)
        dx = float(np.median(movements[:, 0]))
        dy = float(np.median(movements[:, 1]))
        return [dx, dy], good_new

    def process_frame(self, frame):
        """
        Stateful per-frame camera movement estimate (for the single-pass pipeline).
        Updates internal state and returns [dx, dy] for this frame.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movement, good_new = self._estimate_movement(self._old_gray, frame_gray, self._old_features)
        self._old_gray = frame_gray.copy()
        if good_new is not None and len(good_new) >= 4:
            self._old_features = good_new.reshape(-1, 1, 2).astype(np.float32)
        else:
            self._old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
        return movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames_generator, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0.0, 0.0]]  # Frame 0 has no movement
        old_gray = self._old_gray
        old_features = self._old_features

        for frame_num, frame in enumerate(frames_generator):
            if frame_num == 0:
                continue  # First frame already consumed in __init__

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            movement, good_new = self._estimate_movement(old_gray, frame_gray, old_features)
            camera_movement.append(movement)

            old_gray = frame_gray.copy()
            if good_new is not None and len(good_new) >= 4:
                old_features = good_new.reshape(-1, 1, 2).astype(np.float32)
            else:
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames_generator, camera_movement_per_frame):
        cum_x, cum_y = 0.0, 0.0
        for frame_num, frame in enumerate(frames_generator):
            frame = frame.copy()

            dx, dy = camera_movement_per_frame[frame_num]
            cum_x += dx
            cum_y += dy

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(frame, f"Camera Pan X: {cum_x:.0f} px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Pan Y: {cum_y:.0f} px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            yield frame