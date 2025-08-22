"""Video tools."""

from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from PIL import Image

from .logger import get_logger

# Initialize a logger
logger = get_logger(__name__)


def load_video(path: str | Path, as_array: bool = False) -> list[Image.Image] | np.ndarray:
    """Load video as list of PIL images or as a NumPy array.

    Args:
        path (str | Path): The path to the video file.
        as_array (bool, optional): Whether to load the video as a NumPy array. Defaults to False.

    Returns:
        list[Image.Image] | np.ndarray: A list of PIL images or a NumPy array if `as_array` is True. The dimensions
            of the array are (frames, height, width, channels).

    """
    with imageio.get_reader(path) as reader:
        frames = np.stack(list(iter(reader)))

    if as_array:
        return frames

    return [Image.fromarray(frame) for frame in frames]


def save_video(video_frames: list[Image.Image], path: str | Path = "output.mp4", fps: float = 30.0) -> None:
    """Save a list of PIL images to a video file.

    Args:
        video_frames (list[Image.Image]): A list of PIL images to be saved as a video.
        path (str | Path, optional): The path to save the video file. Defaults to "output.mp4".
        fps (float, optional): The frames per second (fps) of the video. Defaults to 30.

    """
    logger.debug("Saving video to %s...", path)
    # imageio expects uint8 images, so we convert from PIL images to numpy arrays
    with imageio.get_writer(path, fps=fps, format="FFMPEG") as writer:
        for frame in video_frames:
            writer.append_data(np.array(frame))
    logger.debug("Video saved successfully.")


def get_video_fps(video_path: str | Path) -> float:
    """Get the frames per second of a video file."""
    video_path = Path(video_path)
    if not video_path.exists():
        msg = f"Video file {video_path} not found."
        raise FileNotFoundError(msg)

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    reader.close()
    return fps
