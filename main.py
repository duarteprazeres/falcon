import argparse
import pickle
import numpy as np
import cv2
import os
from tqdm import tqdm

import supervision as sv

from utils import get_video_generator, get_video_info, save_video_with_sink
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from pitch_detector import PitchDetector
from pitch_visualizer import build_radar_overlay
from ball_tracker import BallKalmanTracker

def _filter_short_tracks(tracks_players, min_frames=8):
    """
    Removes track IDs that appear in fewer than min_frames frames.

    These are spurious detections (ball boys, warmup players near the edge,
    YOLO false positives) that were not caught by the field boundary filter
    and that would otherwise pollute team assignment and statistics.
    """
    from collections import Counter
    counts = Counter(
        tid
        for frame_data in tracks_players
        for tid in frame_data
    )
    to_drop = {tid for tid, n in counts.items() if n < min_frames}
    if not to_drop:
        return 0
    for frame_data in tracks_players:
        for tid in list(frame_data):
            if tid in to_drop:
                del frame_data[tid]
    print(f"  Short-track filter: removed {len(to_drop)} IDs with <{min_frames} frames.")
    return len(to_drop)


def _merge_fragmented_tracks(tracks_players, track_colors, max_gap=120, max_color_dist=45):
    """
    Merges track fragments that likely belong to the same player.

    A track B is merged INTO track A when:
      • A ends strictly before B starts (no temporal overlap)
      • The gap between them is ≤ max_gap frames
      • Their mean jersey BGR colours are within max_color_dist (Euclidean)

    One pass is run; call this function in a loop until it returns 0 for chains.

    track_colors: {track_id: np.ndarray shape (3,)} — mean BGR jersey colour.
    """
    # Rebuild per-track first/last frame from current state
    profiles = {}
    for frame_num, frame_data in enumerate(tracks_players):
        for tid in frame_data:
            if tid not in profiles:
                profiles[tid] = {'first': frame_num, 'last': frame_num}
            else:
                profiles[tid]['last'] = frame_num

    sorted_ids = sorted(profiles, key=lambda t: profiles[t]['first'])
    replacements = {}

    for i, tid_a in enumerate(sorted_ids):
        if tid_a in replacements:
            continue
        prof_a  = profiles[tid_a]
        color_a = track_colors.get(tid_a)
        if color_a is None:
            continue

        for tid_b in sorted_ids[i + 1:]:
            if tid_b in replacements:
                continue
            prof_b = profiles[tid_b]

            # Sorted by first_frame; stop when no candidate can fit in the gap
            if prof_b['first'] > prof_a['last'] + max_gap:
                break

            gap = prof_b['first'] - prof_a['last']
            if gap < 1:
                continue  # overlapping — different players in the same frame

            color_b = track_colors.get(tid_b)
            if color_b is None:
                continue

            if float(np.linalg.norm(color_a - color_b)) < max_color_dist:
                replacements[tid_b] = tid_a
                # Extend A's window so subsequent candidates are evaluated correctly
                if prof_b['last'] > prof_a['last']:
                    profiles[tid_a]['last'] = prof_b['last']
                # Blend B's colour into A for future comparisons
                track_colors[tid_a] = (color_a + color_b) / 2.0
                break

    if not replacements:
        return 0

    for frame_data in tracks_players:
        for tid_from in list(frame_data):
            if tid_from in replacements and tid_from in frame_data:
                tid_to = replacements[tid_from]
                if tid_to not in frame_data:
                    frame_data[tid_to] = frame_data.pop(tid_from)
                else:
                    del frame_data[tid_from]

    return len(replacements)


def _run_combined_detection_pass(
    input_path, stride, tracker, camera_estimator, pitch_detector, total_frames,
    stub_tracks='stubs/track_stubs.pkl',
    stub_camera='stubs/camera_movement_stub.pkl',
    stub_pitch='stubs/pitch_keypoints_stub.pkl',
    skip_pitch_detection=False,
):
    """
    Reads the video exactly ONCE and runs tracking, camera movement estimation,
    and pitch keypoint detection in the same loop.

    Previously these were three separate passes over the full video.
    Combining them cuts video decode time by ~3x.

    Stubs are still honoured: if all three are cached the video is not read at all.
    If any stub is missing the full combined pass is executed and all stubs are saved.
    """
    all_stubs_exist = (
        os.path.exists(stub_tracks)
        and os.path.exists(stub_camera)
        and (skip_pitch_detection or os.path.exists(stub_pitch))
    )

    if all_stubs_exist:
        print("Loading cached detection stubs...")
        with open(stub_tracks, 'rb') as f:
            tracks = pickle.load(f)
        with open(stub_camera, 'rb') as f:
            camera_movement = pickle.load(f)
        with open(stub_pitch, 'rb') as f:
            pitch_keypoints = pickle.load(f)
        return tracks, camera_movement, pitch_keypoints

    print("\nPass 1 (combined): Tracking + Camera Movement + Pitch Keypoints...")

    tracks = {"players": [], "referees": [], "ball": []}
    camera_movement = []
    pitch_keypoints = []

    batch = []
    batch_start = 0

    _BOTSORT_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "botsort_veo.yaml")

    def _flush_batch(batch_frames, start_idx):
        # BoT-SORT with Camera Motion Compensation (gmc_method: sparseOptFlow).
        # persist=True keeps the tracker alive between batch calls so CMC can
        # compute optical flow between consecutive frames across batch boundaries.
        results = tracker.model.track(
            batch_frames,
            conf=0.4,
            imgsz=1280,
            verbose=False,
            device=tracker.device,
            half=True,
            persist=True,
            tracker=_BOTSORT_CFG,
        )

        for i, result in enumerate(results):
            frame_idx = start_idx + i
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            if result.boxes is None or len(result.boxes) == 0:
                continue

            cls_names = result.names
            cls_ids   = result.boxes.cls.int().cpu().tolist()
            confs     = result.boxes.conf.cpu().tolist()
            bboxes    = result.boxes.xyxy.cpu().numpy()

            # — Players and referees (tracked by BoT-SORT) —
            if result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                for tid, cls_id, bbox in zip(ids, cls_ids, bboxes):
                    cls_name = cls_names[cls_id]
                    if cls_name == "goalkeeper":
                        cls_name = "player"
                    if cls_name == "player":
                        tracks["players"][frame_idx][tid] = {"bbox": bbox.tolist()}
                    elif cls_name == "referee":
                        tracks["referees"][frame_idx][tid] = {"bbox": bbox.tolist()}

            # — Ball: highest-confidence detection per frame (interpolated later) —
            best_conf, best_bbox = -1.0, None
            for cls_id, conf, bbox in zip(cls_ids, confs, bboxes):
                if cls_names[cls_id] == "ball" and conf > best_conf:
                    best_conf, best_bbox = conf, bbox
            if best_bbox is not None:
                tracks["ball"][frame_idx][1] = {"bbox": best_bbox.tolist()}

    for frame in tqdm(
        get_video_generator(input_path, stride=stride),
        total=total_frames,
        desc="Detection Pass"
    ):
        # --- Camera movement (stateful, per-frame) ---
        # First frame is consumed in CameraMovementEstimator.__init__, so
        # frame 0 is already accounted for by process_frame returning [0,0].
        if len(camera_movement) == 0:
            camera_movement.append([0.0, 0.0])
        else:
            camera_movement.append(camera_estimator.process_frame(frame))

        # --- Pitch keypoints (per-frame, skipped when calibration is active) ---
        if not skip_pitch_detection:
            pitch_keypoints.append(pitch_detector.detect(frame))
        else:
            pitch_keypoints.append(None)

        # --- Accumulate batch for YOLO ---
        batch.append(frame)
        if len(batch) == 16:
            _flush_batch(batch, batch_start)
            batch_start += 16
            batch = []

    if batch:
        _flush_batch(batch, batch_start)

    with open(stub_tracks, 'wb') as f:
        pickle.dump(tracks, f)
    with open(stub_camera, 'wb') as f:
        pickle.dump(camera_movement, f)
    with open(stub_pitch, 'wb') as f:
        pickle.dump(pitch_keypoints, f)

    return tracks, camera_movement, pitch_keypoints


def main():
    parser = argparse.ArgumentParser(description="Falcon - Football Computer Vision Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to input Veo video")
    parser.add_argument('--output', type=str, required=True, help="Path to save output MP4")
    parser.add_argument('--device', type=str, default='mps', help="Device to run models on (mps, cuda, cpu)")
    parser.add_argument('--stride', type=int, default=1, help="Process 1 in every N frames. Use 2 for 30fps.")
    args = parser.parse_args()

    if not args.output.lower().endswith('.mp4'):
        args.output += '.mp4'
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs('stubs', exist_ok=True)

    PLAYER_MODEL = 'models/football-player-detection.pt'
    PITCH_MODEL = 'models/football-pitch-detection.pt'

    video_info = get_video_info(args.input)
    video_info.fps = video_info.fps / args.stride
    video_info.total_frames = video_info.total_frames // args.stride
    total_frames = video_info.total_frames

    print(f"Video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f}fps | {total_frames} frames (stride {args.stride})")

    print("Initializing modules...")
    tracker = Tracker(PLAYER_MODEL, device=args.device)
    pitch_detector = PitchDetector(PITCH_MODEL)
    first_frame = next(get_video_generator(args.input, stride=args.stride))
    camera_movement_estimator = CameraMovementEstimator(first_frame)

    view_transformer = ViewTransformer()
    calibration_active = view_transformer.load_calibration()
    speed_and_distance_estimator = SpeedAndDistance_Estimator(frame_rate=video_info.fps)
    team_assigner = TeamAssigner(device=args.device)
    player_assigner = PlayerBallAssigner()

    # ----------------------------------------------------------------------
    # PASS 1: Single video read — tracking + camera movement + pitch keypoints
    # If a manual calibration exists the expensive YOLO pitch detection is skipped.
    # ----------------------------------------------------------------------
    tracks, camera_movement_per_frame, pitch_keypoints_per_frame = _run_combined_detection_pass(
        input_path=args.input,
        stride=args.stride,
        tracker=tracker,
        camera_estimator=camera_movement_estimator,
        pitch_detector=pitch_detector,
        total_frames=total_frames,
        skip_pitch_detection=calibration_active,
    )

    # Foot/center positions in pixel space
    tracker.add_position_to_tracks(tracks)

    # Remove players detected outside the pitch (warmup, ball boys, etc.)
    # In calibration mode keypoints are all None, so we use the calibration
    # landmark hull as a static field boundary for every frame.
    print("\nFiltering out-of-field detections...")
    cal_hull = view_transformer.get_calibration_pixel_hull() if calibration_active else None
    if cal_hull is not None:
        print("  Using calibration landmark hull as field boundary.")
    tracker.filter_players_outside_field(tracks, pitch_keypoints_per_frame, static_hull=cal_hull)

    # Adjust pixel positions for camera pan/tilt
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Dynamic homography: pixel → real-world cm
    print("\nComputing dynamic perspective transform...")
    cum_dx, cum_dy = 0.0, 0.0
    for frame_num, keypoints in tqdm(enumerate(pitch_keypoints_per_frame), total=total_frames, desc="Homography"):
        if calibration_active:
            # Accumulate camera panning offset from the reference frame
            dx, dy = camera_movement_per_frame[frame_num]
            cum_dx += dx
            cum_dy += dy
            view_transformer.update_from_camera_movement(cum_dx, cum_dy)
        else:
            view_transformer.update(keypoints)

        for obj in tracks:
            for track_id, track_info in tracks[obj][frame_num].items():
                position = track_info.get('position_adjusted')
                if position is not None:
                    pos_np = np.array(position, dtype=np.float32)
                    transformed = view_transformer.transform_point(pos_np)
                    track_info['position_transformed'] = transformed.tolist() if transformed is not None else None
                else:
                    track_info['position_transformed'] = None

    # Ball tracking: Kalman filter replaces the pandas linear interpolation.
    # Predicts ball position for up to MAX_GAP consecutive missing frames,
    # compensating for camera pan so the prediction stays on the pitch.
    print("\nRunning Kalman filter on ball track...")
    ball_kf = BallKalmanTracker()
    n_predicted = 0
    for frame_num, frame_data in enumerate(tracks["ball"]):
        detected_bbox = frame_data.get(1, {}).get("bbox")
        cam_dx, cam_dy = camera_movement_per_frame[frame_num]
        result = ball_kf.update(detected_bbox, cam_dx, cam_dy)

        if detected_bbox is None and result is not None:
            # Kalman predicted a position — synthesise a 1px bbox at the prediction
            cx, cy = result
            tracks["ball"][frame_num][1] = {"bbox": [cx - 5, cy - 5, cx + 5, cy + 5]}
            n_predicted += 1

    print(f"  Ball: Kalman filled {n_predicted} missing frames "
          f"(out of {sum(1 for f in tracks['ball'] if not f)} undetected).")

    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Diagnostic: report homography coverage to help users debug radar issues
    _n_players_total = sum(len(f) for f in tracks['players'])
    _n_transformed   = sum(
        1 for f in tracks['players'] for info in f.values()
        if info.get('position_transformed') is not None
    )
    if _n_players_total > 0:
        _pct = 100 * _n_transformed / _n_players_total
        print(f"  Homography coverage: {_n_transformed}/{_n_players_total} player-frames ({_pct:.0f}%)")
        if _pct < 20:
            print("  ⚠  Low coverage — radar will be mostly empty.")
            print("     In calibration mode: verify the calibration.json is correct (python calibrate.py).")
            print("     In YOLO mode: the pitch model may not detect enough keypoints for this camera angle.")
        elif _pct < 60:
            print("  ⚠  Partial coverage — some frames lack a valid homography (camera too zoomed in).")
        # Log sample coordinates from the first frame that has valid transforms
        for f in tracks['players']:
            samples = [(pid, info['position_transformed'])
                       for pid, info in f.items()
                       if info.get('position_transformed') is not None][:3]
            if samples:
                sample_str = ', '.join(f"pid {pid}: ({x:.0f}, {y:.0f})" for pid, (x, y) in samples)
                print(f"  Sample world coords (cm): {sample_str}")
                break

    # ----------------------------------------------------------------------
    # PASS 1b (partial): Sample crops for team fit + collect jersey colours
    # per track_id (used for track merging below).
    # Reads frames 0–min(300, total) every ~15 frames so both teams appear.
    # ----------------------------------------------------------------------
    _sample_limit = min(300, total_frames)
    _sample_step  = max(1, _sample_limit // 20)  # ~20 sample points
    print(f"\nSampling in-field crops (frames 0–{_sample_limit}, step {_sample_step})...")
    sample_crops   = []
    track_colors   = {}   # {track_id: [color_array, ...]}  for track merging

    for i, frame in enumerate(get_video_generator(args.input, stride=args.stride)):
        if i >= _sample_limit:
            break
        for pid, track in tracks['players'][i].items():
            crop = team_assigner.get_crop(frame, track['bbox'])
            # Always extract colour (for merger), only sample every _step for fit
            color = team_assigner._extract_jersey_color(crop)
            if color is not None:
                track_colors.setdefault(pid, []).append(color)
            if i % _sample_step == 0 and crop.size > 0:
                sample_crops.append(crop)

    # Mean colour per track
    track_mean_colors = {
        tid: np.mean(cols, axis=0)
        for tid, cols in track_colors.items()
    }

    team_assigner.fit_from_crops(sample_crops)

    # ----------------------------------------------------------------------
    # POST-PROCESSING: remove spurious short tracks, then merge fragments
    # Must happen before Pass 2 so team assignment sees fewer, cleaner IDs.
    # ----------------------------------------------------------------------
    n_ids_before = len({tid for f in tracks['players'] for tid in f})
    print(f"\nPost-processing tracks ({n_ids_before} unique IDs)...")

    _filter_short_tracks(tracks['players'], min_frames=8)

    print("  Merging fragmented tracks (same colour, gap ≤ 120 frames)...")
    total_merged = 0
    for _pass in range(6):
        n = _merge_fragmented_tracks(tracks['players'], track_mean_colors,
                                     max_gap=120, max_color_dist=45)
        total_merged += n
        if n == 0:
            break

    n_ids_after = len({tid for f in tracks['players'] for tid in f})
    print(f"  IDs: {n_ids_before} → {n_ids_after}  ({total_merged} merges)")

    # ----------------------------------------------------------------------
    # PASS 2: Team assignment + ball control (full video read)
    # Must happen before drawing so team colours are stored in tracks[] before
    # draw_annotations reads them from there.
    # ----------------------------------------------------------------------
    print("\nPass 2: Team assignment & ball control...")
    team_ball_control = []

    for frame_num, frame in tqdm(
        enumerate(get_video_generator(args.input, stride=args.stride)),
        total=total_frames,
        desc="Team Assign"
    ):
        player_track = tracks['players'][frame_num]

        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame, track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        ball_bbox = tracks['ball'][frame_num][1]['bbox'] if tracks['ball'][frame_num] else []
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)

    # ----------------------------------------------------------------------
    # PASS 3: Drawing & video export (streaming, single video read)
    # ----------------------------------------------------------------------
    print("\nPass 3: Drawing annotations and saving MP4...")

    with save_video_with_sink(get_video_generator(args.input, stride=args.stride), video_info, args.output) as sink:

        annotated_frames = tracker.draw_annotations(
            get_video_generator(args.input, stride=args.stride),
            tracks,
            team_ball_control,
        )
        annotated_frames = camera_movement_estimator.draw_camera_movement(
            annotated_frames, camera_movement_per_frame
        )
        annotated_frames = speed_and_distance_estimator.draw_speed_and_distance(
            annotated_frames, tracks
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
                
    # ----------------------------------------------------------------------
    # CSV EXPORT — frame-level tracking data for analysis
    # ----------------------------------------------------------------------
    import csv
    csv_path = os.path.splitext(args.output)[0] + "_tracks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class_name",
                    "x_center", "y_center", "team",
                    "speed_kmh", "distance_m",
                    "world_x_cm", "world_y_cm"])
        for frame_num in range(total_frames):
            for tid, info in tracks["players"][frame_num].items():
                b = info["bbox"]
                xc, yc = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                pt = info.get("position_transformed") or ["", ""]
                w.writerow([frame_num, tid, "player",
                            round(xc, 1), round(yc, 1),
                            info.get("team", ""),
                            round(info["speed"], 2) if "speed" in info else "",
                            round(info["distance"], 2) if "distance" in info else "",
                            round(pt[0], 1) if pt[0] != "" else "",
                            round(pt[1], 1) if pt[1] != "" else ""])
            for tid, info in tracks["ball"][frame_num].items():
                b = info["bbox"]
                xc, yc = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                w.writerow([frame_num, tid, "ball",
                            round(xc, 1), round(yc, 1),
                            "", "", "", "", ""])
    print(f"  CSV saved: {csv_path}")

    print(f"\n✅ Pipeline complete! Output saved to: {args.output}")

if __name__ == '__main__':
    main()