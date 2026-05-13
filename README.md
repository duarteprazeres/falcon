# Falcon — Football Computer Vision Pipeline

Falcon is a computer vision scouting pipeline built for Veo broadcast footage. It detects and tracks players, referees, and the ball frame-by-frame, assigns players to teams by jersey colour, estimates real-world positions via homography, and overlays a live radar on the output video.

Designed for a moving auto-following camera (constant zoom/pan), not a fixed tactical feed.

## Features

- **Player & ball detection** — YOLOv11 models fine-tuned on football footage (players, referees, ball, pitch keypoints)
- **Multi-object tracking** — BoT-SORT with Camera Motion Compensation (`sparseOptFlow`) to handle Veo's constant panning and zooming
- **Team assignment** — Corner-trick KMeans jersey colour extraction; ID-independent (works even when the tracker resets IDs)
- **Ball Kalman filter** — Constant-velocity Kalman filter predicts ball position for up to 10 consecutive missing frames, compensating for camera movement
- **Track post-processing** — Short-track filter removes spurious IDs; colour-based merger reassembles fragmented tracks from zoom events
- **Dynamic homography** — Pixel → real-world cm transform updated per frame from YOLO pitch keypoints, or from a manual calibration file when keypoints are unavailable
- **Speed & distance estimation** — Per-player speed (km/h) and total distance covered (m), computed in real-world coordinates
- **Live radar overlay** — Top-down pitch minimap rendered on every frame with team-coloured player dots
- **CSV export** — Frame-level tracking data (player ID, team, speed, world coordinates) saved alongside the output video

## Pipeline Overview

```
Video
  │
  ├─ Pass 1 (single read): YOLO detection + BoT-SORT tracking
  │                        Camera movement (optical flow)
  │                        Pitch keypoint detection
  │
  ├─ Post-processing:      Short-track filter (< 8 frames → spurious)
  │                        Colour-based track merger (gap ≤ 120 frames)
  │
  ├─ Pass 1b (partial):    Jersey colour sampling → KMeans team fit
  │
  ├─ Ball Kalman:          Fill gaps up to 10 frames
  │
  ├─ Homography:           Pixel → real-world cm per frame
  │
  ├─ Pass 2:               Team assignment + ball control
  │
  └─ Pass 3:               Draw annotations + radar → MP4 output
                           CSV export
```

## Models

| Model | Purpose |
|---|---|
| `football-player-detection.pt` | Detects players, referees, ball (YOLOv11, fine-tuned) |
| `football-pitch-detection.pt` | Detects pitch keypoints for homography |

Place models in the `models/` directory. Use `models/download_models.sh` to fetch them.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --input input_videos/match.mp4 --output output_videos/result.mp4 --device mps
```

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to input video |
| `--output` | required | Path for output MP4 |
| `--device` | `mps` | Inference device: `mps`, `cuda`, or `cpu` |
| `--stride` | `1` | Process 1 in every N frames (use `2` for 60fps → 30fps) |

Stubs are saved to `stubs/` after the first run so subsequent runs skip the expensive detection pass.

## Manual Camera Calibration

For footage where YOLO cannot detect enough pitch keypoints (heavily zoomed or unusual angles), run the interactive calibration tool:

```bash
python calibrate.py --input input_videos/match.mp4
```

This saves `calibration/calibration.json` which the pipeline picks up automatically on the next run.

## Output

- **MP4** — annotated video with player ellipses (team coloured), ball triangle, speed labels, ball control %, and radar overlay
- **CSV** — `<output_name>_tracks.csv` with columns: `frame`, `track_id`, `class_name`, `x_center`, `y_center`, `team`, `speed_kmh`, `distance_m`, `world_x_cm`, `world_y_cm`

## Requirements

```
ultralytics
supervision
opencv-python
numpy
pandas
filterpy
scikit-learn
scipy
torch
transformers
umap-learn
tqdm
gdown
matplotlib
git+https://github.com/roboflow/sports.git
```
