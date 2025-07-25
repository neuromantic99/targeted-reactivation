from pathlib import Path
from typing import List, Tuple
import ffmpeg
import numpy as np
import cv2

from rsync import Rsync_aligner


def get_number_of_frames(video_path: Path) -> int:
    return int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_COUNT))


def extract_frames_fast(video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_array = np.zeros((height, width, len(frame_indices), 3), dtype=np.uint8)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Jump to frame
        ret, frame = cap.read()
        if ret:
            frame_array[:, :, i, :] = frame

    cap.release()
    return frame_array


def save_video(
    frames: np.ndarray, output_path: str = "output.mp4", fps: int = 30
) -> None:
    """
    Save a NumPy array of frames as an MP4 video using FFmpeg.

    Args:
        frames (np.ndarray): Video frames of shape (num_frames, height, width, 3).
        output_path (str): Path to save the MP4 file.
        fps (int): Frames per second for the output video.

    Returns:
        None
    """
    _, height, width = frames.shape

    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="gray",
            s=f"{width}x{height}",
            framerate=fps,
        )
        .output(
            output_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=23,
            movflags="faststart",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())

    process.stdin.close()
    process.wait()


def get_data_paths(
    data_folder: Path,
) -> Tuple[List[Path], List[Path], List[Path]]:
    return (
        sorted(list(data_folder.glob("*.mp4"))),
        sorted(list(data_folder.glob("*time.npy"))),
        sorted(list(data_folder.glob("*.tsv"))),
    )


def get_aligners(
    sync_npx: np.ndarray, rsync_times: List[np.ndarray]
) -> List[Rsync_aligner]:
    chunk_start = 0
    # A list of Rsync_aligner objects, one for each session
    # So in theory, aligner 0 is the conditioning aligner.
    aligners = []
    for rsync_time in rsync_times:
        aligners.append(
            Rsync_aligner(
                sync_npx[chunk_start : chunk_start + len(rsync_time)],
                rsync_time,
                raise_exception=True,
            )
        )

        chunk_start += len(rsync_time)

    return aligners
