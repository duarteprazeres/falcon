import argparse
import numpy as np
import cv2
import os
from tqdm import tqdm

from utils import get_video_generator, get_video_info, save_video_with_sink
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from pitch_detector import PitchDetector
from pitch_visualizer import build_radar_overlay

def main():
    parser = argparse.ArgumentParser(description="Falcon - Football Computer Vision Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to input Veo video")
    parser.add_argument('--output', type=str, required=True, help="Path to save output MP4")
    # For M2, we use 'mps', for generic 'cpu' or 'cuda'
    parser.add_argument('--device', type=str, default='mps', help="Device to run models on (mps, cuda, cpu)")
    parser.add_argument('--stride', type=int, default=1, help="Process 1 in every N frames. Use 2 for 30fps.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs('stubs', exist_ok=True)

    # Model paths
    PLAYER_MODEL = 'models/football-player-detection.pt'
    PITCH_MODEL = 'models/football-pitch-detection.pt'

    # Retrieve video info to configure modules
    video_info = get_video_info(args.input)
    # Adjust video info to reflect the stride
    video_info.fps = video_info.fps / args.stride
    video_info.total_frames = video_info.total_frames // args.stride
    total_frames = video_info.total_frames
    
    print(f"Video configured to: {video_info.width}x{video_info.height} @ {video_info.fps:.1f}fps | {video_info.total_frames} frames (Stride: {args.stride})")

    # 1. Initialize all modules
    print("Initializing modules...")
    tracker = Tracker(PLAYER_MODEL, device=args.device)
    pitch_detector = PitchDetector(PITCH_MODEL)
    
    # We need the first frame to initialize CameraMovementEstimator and TeamAssigner
    first_frame = next(get_video_generator(args.input, stride=args.stride))
    camera_movement_estimator = CameraMovementEstimator(first_frame)
    
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=video_info.fps)
    team_assigner = TeamAssigner(device=args.device)
    player_assigner = PlayerBallAssigner()
    
    # ----------------------------------------------------------------------
    # PASS 1: TRACKING & METRICS (Saves to Memory / Stubs)
    # ----------------------------------------------------------------------
    print("\nPass 1a: Object Tracking...")
    tracks = tracker.get_object_tracks(
        tqdm(get_video_generator(args.input, stride=args.stride), total=total_frames, desc="Tracking"), 
        read_from_stub=True, 
        stub_path='stubs/track_stubs.pkl'
    )
    
    print("\nPass 1b: Camera Movement...")
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        tqdm(get_video_generator(args.input, stride=args.stride), total=total_frames, desc="Camera Move"), 
        read_from_stub=True, 
        stub_path='stubs/camera_movement_stub.pkl'
    )
    
    print("\nPass 1c: Pitch Detection (Keypoints)...")
    pitch_keypoints_per_frame = pitch_detector.get_pitch_keypoints(
        tqdm(get_video_generator(args.input, stride=args.stride), total=total_frames, desc="Pitch Keypoints"),
        read_from_stub=True,
        stub_path='stubs/pitch_keypoints_stub.pkl'
    )

    # Set initial bounding box positions (feet/center)
    tracker.add_position_to_tracks(tracks)
    # Adjust tracking positions based on camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Dynamic Perspective Transform (pixels to real-world cm)
    print("\nComputing dynamic perspective transform...")
    for frame_num, keypoints in tqdm(enumerate(pitch_keypoints_per_frame), total=total_frames, desc="Homography"):
        # Update homography matrix for this frame dynamically based on detected lines
        view_transformer.update(keypoints)
        
        # Apply transformation to tracks in this frame
        for obj in tracks:
            for track_id, track_info in tracks[obj][frame_num].items():
                position = track_info.get('position_adjusted')
                if position is not None:
                    pos_np = np.array(position, dtype=np.float32)
                    transformed = view_transformer.transform_point(pos_np)
                    if transformed is not None:
                        track_info['position_transformed'] = transformed.tolist()
                    else:
                        track_info['position_transformed'] = None
                else:
                    track_info['position_transformed'] = None

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # ----------------------------------------------------------------------
    # PASS 1d: TEAM ASSIGNMENT & BALL CONTROL
    # ----------------------------------------------------------------------
    print("\nPass 1d: Team Assignment & Ball Control (SigLIP)...")
    
    # Collect sample crops from multiple frames to train the classifier robustly
    print("Sampling player crops for team clustering...")
    sample_crops = []
    for i, frame in enumerate(get_video_generator(args.input, stride=args.stride)):
        if i >= 60:
            break
        if i % 5 == 0:  # Sample every 5th frame
            for pid, track in tracks['players'][i].items():
                crop = team_assigner.get_crop(frame, track['bbox'])
                if crop.size > 0:
                    sample_crops.append(crop)
    
    team_assigner.fit_from_crops(sample_crops)
    team_ball_control = []
    
    for frame_num, frame in tqdm(enumerate(get_video_generator(args.input, stride=args.stride)), total=total_frames, desc="Team Assign"):
        player_track = tracks['players'][frame_num]
        
        # Assign Teams using SigLIP embeddings
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame, track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Assign Ball Acquisition
        ball_bbox = tracks['ball'][frame_num][1]['bbox'] if tracks['ball'][frame_num] else []
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            
    team_ball_control = np.array(team_ball_control)

    # ----------------------------------------------------------------------
    # PASS 2: DRAWING & VIDEO EXPORT (Streaming output)
    # ----------------------------------------------------------------------
    print("\nPass 2: Drawing annotations and saving MP4...")
    
    with save_video_with_sink(get_video_generator(args.input, stride=args.stride), video_info, args.output) as sink:
        
        # We chain generators to process frames efficiently
        annotated_frames = tracker.draw_annotations(
            get_video_generator(args.input, stride=args.stride), 
            tracks, 
            team_ball_control
        )
        
        annotated_frames = camera_movement_estimator.draw_camera_movement(
            annotated_frames, 
            camera_movement_per_frame
        )
        
        annotated_frames = speed_and_distance_estimator.draw_speed_and_distance(
            annotated_frames, 
            tracks
        )
        
        for frame_num, frame in tqdm(enumerate(annotated_frames), total=total_frames, desc="Exporting MP4"):
            
            # --- RADAR OVERLAY ---
            # Get real-world coordinates for all players in this frame
            player_positions = {
                pid: info.get('position_transformed')
                for pid, info in tracks['players'][frame_num].items()
                if info.get('position_transformed') is not None
            }
            team_assignments = {
                pid: info.get('team')
                for pid, info in tracks['players'][frame_num].items()
            }
            
            if player_positions:
                radar = build_radar_overlay(
                    player_positions=player_positions,
                    team_assignments=team_assignments,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    radar_scale=0.06
                )
                
                # Radar shape is scaled. Overlay it at bottom center
                h, w = frame.shape[:2]
                rh, rw = radar.shape[:2]
                
                x_offset = w // 2 - rw // 2
                y_offset = h - rh - 20
                
                if x_offset > 0 and y_offset > 0 and x_offset + rw < w and y_offset + rh < h:
                    alpha = 0.7
                    roi = frame[y_offset:y_offset+rh, x_offset:x_offset+rw]
                    blended = cv2.addWeighted(roi, 1-alpha, radar, alpha, 0)
                    frame[y_offset:y_offset+rh, x_offset:x_offset+rw] = blended

            sink.write_frame(frame)
                
    print(f"\n✅ Pipeline complete! Output saved to: {args.output}")

if __name__ == '__main__':
    main()