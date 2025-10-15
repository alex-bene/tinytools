"""Video tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image

from tinytools.logger import get_logger

from .file_video_stream import FileVideoStream

# Initialize a logger
logger = get_logger(__name__)


def load_video(
    path: str | Path, max_frames: int = -1, as_array: bool = False, as_rgb: bool = False
) -> list[Image.Image] | np.ndarray:
    """Load video as list of PIL images or as a NumPy array.

    Args:
        path (str | Path): The path to the video file.
        max_frames (int, optional): The maximum number of frames to load. Defaults to -1 (all frames).
        as_array (bool, optional): Whether to load the video as a NumPy array. Defaults to False.
        as_rgb (bool, optional): Whether to load the video as RGB. Only applicable if `as_array` is True.
            Defaults to False.

    Returns:
        list[Image.Image] | np.ndarray: A list of PIL images or a NumPy array if `as_array` is True. The dimensions
            of the array are (frames, height, width, channels).

    """
    if not Path(path).exists():
        return [] if not as_array else np.empty((0, 0, 0, 3))

    transform = None
    if not as_array:
        transform = lambda x: Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))  # noqa: E731
    elif as_rgb:  # and as_array
        transform = partial(cv2.cvtColor, code=cv2.COLOR_BGR2RGB)

    cap = FileVideoStream(str(path), transform=transform, max_frames=max_frames)
    cap.start()
    frames = []
    while cap.running():
        frames.append(cap.read())
    cap.stop()

    return np.stack(frames) if as_array else frames


def load_videos(
    paths: list[str | Path], max_frames: int = -1, as_array: bool = False, as_rgb: bool = False, workers: int = 32
) -> list[list[Image.Image] | np.ndarray]:
    """Load multiple videos in parallel as a list of lists of PIL images or as a list of NumPy arrays.

    Args:
        paths (list[str | Path]): The paths to the video files.
        max_frames (int, optional): The maximum number of frames to load. Defaults to -1 (all frames).
        as_array (bool, optional): Whether to load the video as a NumPy array. Defaults to False.
        as_rgb (bool, optional): Whether to load the video as RGB. Only applicable if `as_array` is True.
            Defaults to False.
        workers (int, optional): The maximum number of threads to use for parallel loading. Defaults to 32.

    Returns:
        list[list[Image.Image] | np.ndarray]: A list of lists of PIL images or a list of NumPy arrays if `as_array` is
            True. The dimensions of the array are (frames, height, width, channels).

    """
    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as ex:
        future_to_idx = {
            ex.submit(load_video, path=p, max_frames=max_frames, as_array=as_array, as_rgb=as_rgb): i
            for i, p in enumerate(paths)
        }
        for fut in as_completed(future_to_idx):
            results[future_to_idx[fut]] = fut.result()

    return results


def save_video(video_frames: list[Image.Image], path: str | Path = "output.mp4", fps: float = 30.0) -> None:
    """Save a list of PIL images to a video file.

    Args:
        video_frames (list[Image.Image]): A list of PIL images to be saved as a video.
        path (str | Path, optional): The path to save the video file. Defaults to "output.mp4".
        fps (float, optional): The frames per second (fps) of the video. Defaults to 30.

    """
    logger.debug("Saving video to %s...", path)
    # imageio expects uint8 images, so we convert from PIL images to numpy arrays
    # TODO: there is also imageio.mimsave
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
