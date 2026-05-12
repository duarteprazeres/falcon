import cv2
from utils import measure_distance, get_foot_position

class SpeedAndDistance_Estimator():
    def __init__(self, frame_rate=60):
        self.frame_window = 5
        self.frame_rate = frame_rate
        self.max_speed = 38.0 # Human sprint max is ~38 km/h. Anything above is a tracking artifact.
    
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue 
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id].get('position_transformed')
                    end_position = object_tracks[last_frame][track_id].get('position_transformed')

                    if start_position is None or end_position is None:
                        continue
                    
                    # New ViewTransformer outputs in centimetres, so we divide by 100 to get metres
                    distance_covered_cm = measure_distance(start_position, end_position)
                    distance_covered_m = distance_covered_cm / 100.0

                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered_m / time_elapsed if time_elapsed > 0 else 0
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Cap speed and ignore distance jumps that are physically impossible
                    if speed_km_per_hour > self.max_speed:
                        speed_km_per_hour = 0
                        distance_covered_m = 0

                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    
                    total_distance[object][track_id] += distance_covered_m

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
    
    def draw_speed_and_distance(self, frames_generator, tracks):
        for frame_num, frame in enumerate(frames_generator):
            frame = frame.copy()
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue 
                for _, track_info in object_tracks[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed', None)
                       distance = track_info.get('distance', None)
                       if speed is None or distance is None:
                           continue
                       
                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1] += 40

                       position = tuple(map(int, position))
                       cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                       cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            
            yield frame