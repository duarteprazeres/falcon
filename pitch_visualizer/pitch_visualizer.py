"""
Pitch visualizer module for the Falcon project.

Wraps the roboflow/sports soccer annotators to draw player positions,
trajectories, and team zones on a top-down pitch diagram.

These visualizations are the foundation for:
- Player heatmaps (accumulate positions over time → gaussian blur)
- Radar / mini-map overlays on the output video
- Team control zones (Voronoi diagrams)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import supervision as sv

from sports.annotators.soccer import (
    draw_pitch as _draw_pitch,
    draw_points_on_pitch as _draw_points_on_pitch,
    draw_paths_on_pitch as _draw_paths_on_pitch,
    draw_pitch_voronoi_diagram,
)
from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()

# Default team colors (consistent with tracker ellipse colors)
TEAM_1_COLOR = sv.Color.from_hex('#FF1493')  # Pink — Team 1
TEAM_2_COLOR = sv.Color.from_hex('#00BFFF')  # Blue — Team 2
REFEREE_COLOR = sv.Color.from_hex('#FFD700')  # Gold — Referees
BALL_COLOR = sv.Color.from_hex('#00FF00')      # Green — Ball


def draw_pitch(
    background_color: sv.Color = sv.Color(34, 139, 34),
    scale: float = 0.1,
    padding: int = 50,
) -> np.ndarray:
    """
    Draws a blank soccer pitch diagram.

    Args:
        background_color (sv.Color): Pitch green background.
        scale (float): Scale factor. 0.1 means 1px = 10cm.
        padding (int): Padding in pixels around the pitch.

    Returns:
        np.ndarray: BGR image of the pitch.
    """
    return _draw_pitch(
        config=CONFIG,
        background_color=background_color,
        scale=scale,
        padding=padding,
    )


def draw_players_on_pitch(
    player_positions: Dict[int, Tuple[float, float]],
    team_assignments: Dict[int, int],
    team_colors: Optional[Dict[int, sv.Color]] = None,
    pitch: Optional[np.ndarray] = None,
    radius: int = 16,
    scale: float = 0.1,
    padding: int = 50,
) -> np.ndarray:
    """
    Draws player positions on the pitch diagram, colored by team.

    Args:
        player_positions: {player_id: (x_cm, y_cm)} — real-world coordinates in cm.
        team_assignments: {player_id: team_id} — 1 or 2.
        team_colors: Optional override for team colors. Default: pink/blue.
        pitch: Existing pitch image to draw on. Created fresh if None.
        radius (int): Dot radius in pixels.
        scale (float): Must match the scale used in draw_pitch().
        padding (int): Must match the padding used in draw_pitch().

    Returns:
        np.ndarray: Pitch image with player dots drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(scale=scale, padding=padding)

    if team_colors is None:
        team_colors = {1: TEAM_1_COLOR, 2: TEAM_2_COLOR}

    for team_id, color in team_colors.items():
        positions = np.array([
            pos for pid, pos in player_positions.items()
            if team_assignments.get(pid) == team_id and pos is not None
        ], dtype=np.float32)

        if len(positions) > 0:
            pitch = _draw_points_on_pitch(
                config=CONFIG,
                xy=positions,
                face_color=color,
                radius=radius,
                scale=scale,
                padding=padding,
                pitch=pitch,
            )

    return pitch


def draw_player_paths_on_pitch(
    player_paths: Dict[int, List[Tuple[float, float]]],
    team_assignments: Dict[int, int],
    team_colors: Optional[Dict[int, sv.Color]] = None,
    pitch: Optional[np.ndarray] = None,
    thickness: int = 2,
    scale: float = 0.1,
    padding: int = 50,
) -> np.ndarray:
    """
    Draws player movement trajectories on the pitch (lines connecting positions).

    This is the foundation for player tracking paths and heatmap generation.

    Args:
        player_paths: {player_id: [(x1, y1), (x2, y2), ...]} — ordered real-world positions.
        team_assignments: {player_id: team_id}
        team_colors: Optional override for team colors.
        pitch: Existing pitch image. Created fresh if None.
        thickness (int): Line thickness in pixels.
        scale (float): Must match draw_pitch() scale.
        padding (int): Must match draw_pitch() padding.

    Returns:
        np.ndarray: Pitch image with trajectory lines drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(scale=scale, padding=padding)

    if team_colors is None:
        team_colors = {1: TEAM_1_COLOR, 2: TEAM_2_COLOR}

    for team_id, color in team_colors.items():
        paths = [
            np.array(path, dtype=np.float32)
            for pid, path in player_paths.items()
            if team_assignments.get(pid) == team_id and len(path) >= 2
        ]

        if paths:
            pitch = _draw_paths_on_pitch(
                config=CONFIG,
                paths=paths,
                color=color,
                thickness=thickness,
                scale=scale,
                padding=padding,
                pitch=pitch,
            )

    return pitch


def build_radar_overlay(
    player_positions: Dict[int, Tuple[float, float]],
    team_assignments: Dict[int, int],
    frame_width: int,
    frame_height: int,
    radar_scale: float = 0.06,
    opacity: float = 0.7,
) -> np.ndarray:
    """
    Builds a semi-transparent radar mini-map to overlay on the video frame.

    This replicates the RADAR mode from roboflow/sports, adapted to use the
    Falcon tracking data format.

    Args:
        player_positions: {player_id: (x_cm, y_cm)} — real-world cm coordinates.
        team_assignments: {player_id: team_id}
        frame_width (int): Width of the video frame (for sizing the radar).
        frame_height (int): Height of the video frame.
        radar_scale (float): Scale factor for the mini-map. Smaller = smaller radar.
        opacity (float): Opacity of the radar overlay (0=transparent, 1=opaque).

    Returns:
        np.ndarray: Radar image ready to overlay on a video frame.
    """
    # Clamp coordinates to pitch boundaries to prevent points drawing off the radar image
    valid_positions = {}
    for pid, pos in player_positions.items():
        if pos is not None:
            x, y = pos
            x = max(0, min(12000, x))
            y = max(0, min(7000, y))
            valid_positions[pid] = (x, y)

    pitch = draw_pitch(scale=radar_scale)
    pitch = draw_players_on_pitch(
        player_positions=valid_positions,
        team_assignments=team_assignments,
        pitch=pitch,
        radius=10,
        scale=radar_scale,
    )
    return pitch
