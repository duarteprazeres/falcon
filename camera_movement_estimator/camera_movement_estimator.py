import pickle
import cv2
import numpy as np
import os
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 1.0

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        h, w = first_frame_grayscale.shape
        mask_features = np.zeros_like(first_frame_grayscale)
        
        # Mask the borders of the image to track static background (sky, trees, stadium walls)
        # instead of the pitch where players are moving.
        mask_features[0:h//10, :] = 1          # Top 10%
        mask_features[h - h//10:, :] = 1       # Bottom 10%
        mask_features[:, 0:w//10] = 1          # Left 10%
        mask_features[:, w - w//10:] = 1       # Right 10%

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    

    def get_camera_movement(self, frames_generator, read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = []
        old_gray = None
        old_features = None

        for frame_num, frame in enumerate(frames_generator):
            frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            if old_gray is None:
                old_gray = frame_gray
                old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)
                camera_movement.append([0,0])
                continue

            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement.append([camera_movement_x,camera_movement_y])
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
            else:
                camera_movement.append([0,0])

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self, frames_generator, camera_movement_per_frame):
        for frame_num, frame in enumerate(frames_generator):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            yield frame 