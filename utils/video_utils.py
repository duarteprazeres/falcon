import cv2
import supervision as sv
from typing import Iterator
import numpy as np


def get_video_generator(video_path: str, stride: int = 1) -> Iterator[np.ndarray]:
    """
    Yields frames from a video file one at a time (streaming).

    This is the preferred function for processing video — it does NOT load the
    entire video into RAM, making it safe for long Veo recordings (90+ minutes).

    Args:
        video_path (str): Path to the input video file.
        stride (int): Process every Nth frame. Default=1 (every frame).

    Yields:
        np.ndarray: Individual video frames in BGR format.
    """
    return sv.get_video_frames_generator(source_path=video_path, stride=stride)


def get_video_info(video_path: str) -> sv.VideoInfo:
    """
    Returns metadata about a video (resolution, fps, total frames).

    Args:
        video_path (str): Path to the video file.

    Returns:
        sv.VideoInfo: Object with .width, .height, .fps, .total_frames attributes.
    """
    return sv.VideoInfo.from_video_path(video_path)


def save_video(output_video_frames: list, output_video_path: str) -> None:
    """
    Saves a list of annotated frames to a video file (MP4/H.264).

    Args:
        output_video_frames (list): List of numpy frame arrays.
        output_video_path (str): Destination path. Use .mp4 extension.
    """
    if not output_video_frames:
        return

    h, w = output_video_frames[0].shape[:2]
    # Use MP4 with H.264 for broad compatibility (replaces the old XVID .avi)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def save_video_with_sink(
    frame_generator: Iterator[np.ndarray],
    video_info: sv.VideoInfo,
    output_video_path: str
) -> sv.VideoSink:
    """
    Returns an sv.VideoSink context manager for streaming frame-by-frame output.

    Usage:
        video_info = get_video_info(source_path)
        with save_video_with_sink(video_info, 'output.mp4') as sink:
            for frame in get_video_generator(source_path):
                annotated = annotate(frame)
                sink.write_frame(annotated)

    Args:
        frame_generator: Not used directly — kept for signature clarity.
        video_info (sv.VideoInfo): Video metadata from get_video_info().
        output_video_path (str): Destination path.

    Returns:
        sv.VideoSink: Context manager that writes frames to disk.
    """
    # Use avc1 (H.264) codec so it opens natively on Mac/QuickTime
    return sv.VideoSink(target_path=output_video_path, video_info=video_info, codec="avc1")


# ---------------------------------------------------------------------------
# DEPRECATED — kept only for stub-based development workflows
# ---------------------------------------------------------------------------

def read_video(video_path: str) -> list:
    """
    DEPRECATED: Loads the entire video into RAM as a list of frames.

    ⚠️  Do NOT use for long videos (> a few minutes) — it will exhaust RAM.
    This function is kept only to support the existing .pkl stub cache system
    during development. For production, use get_video_generator() instead.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: All video frames as numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
