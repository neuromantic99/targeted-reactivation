from pathlib import Path
from typing import List, Tuple
import ffmpeg
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd

from data_import import Session
from models import LED, Sound
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


def process_session(session: Session) -> Tuple[List[Sound], List[LED] | None]:
    sound_prints = [
        printed
        for printed in session.prints
        if printed.string.startswith("Deliverying sound frequency")
    ]
    sounds = [
        Sound(sound.time, int(sound.string.split("Deliverying sound frequency ")[1]))
        for sound in sound_prints
    ]

    if session.task_name == "sleeping_alex":
        return sounds, None

    led_prints = [
        printed
        for printed in session.prints
        if printed.string.startswith("Turning on LED Color")
    ]

    leds = [
        LED(led.time, led.string.split("Turning on LED Color: ")[1])
        for led in led_prints
    ]
    # Session stopped in between the two stims
    if len(leds) - len(sounds) == 1:
        # Remove the last LED
        # TODO: add a check that they pair correctly
        leds = leds[:-1]

    for sound, led in zip(sounds, leds, strict=True):
        # Should be image then audio
        assert led.time < sound.time

        assert 0.99 < sound.time - led.time < 1.01

    return sounds, leds


def process_sleep_spreadsheet(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    num_to_state = {0.5: "nrem", 0: "deep nrem", 1: "rem", 2: "awake", 4: "movement"}

    mouse = data_path.parts[-1]

    spreadsheet_path = Path(
        "/Volumes/MarcBusche/Alex/Reactivations/Sleep Scoring/results"
    )

    spreadsheets = list(spreadsheet_path.glob(f"*.xlsx"))
    mouse_sheets = [
        file_name
        for file_name in spreadsheets
        if mouse in str(file_name).lower()
        and "tones" not in str(file_name).lower()
        and file_name.name[:2] != "~$"  # Exclude temporary files
    ]
    assert (
        len(mouse_sheets) == 1
    ), f"Expected one sleep scoring spreadsheet for mouse {mouse}, found {len(mouse_sheets)}."
    spreadsheet = mouse_sheets[0]

    data = pd.read_excel(spreadsheet, sheet_name="Sheet1")

    mins = data["Minutes"].to_numpy()
    seconds = data["Seconds"].to_numpy()
    score = data["Score"].to_numpy()

    # Data missing at the end of the spreadsheet, had a look at the video. Mouse is moving a lot so assigned to awake
    if "10681" in str(data_path):
        first_nan = np.where(np.isnan(score))[0][0]
        assert first_nan > 29 * 60
        assert np.all(np.isnan(score[first_nan:]))
        score[first_nan:] = 2  # awake

    total_seconds = mins * 60 + seconds

    # Mistake in all spreadsheets where the minute is set to 5 when it should be 6
    total_seconds[368] = 368
    assert np.all(np.diff(total_seconds) == 1)

    return total_seconds, np.array([num_to_state[state] for state in score])


def get_lfp_index_sleep_state(
    data_folder: Path,
    n_samples: int,
    sampling_rate_lfp: float,
    plot: bool = False,
) -> dict[str, np.ndarray]:
    """Havent properly tested this yet, but the hacky plot looks fine"""
    seconds, sleep_state = process_sleep_spreadsheet(data_folder)

    assert abs(seconds[-1] - n_samples / sampling_rate_lfp) < 1.5

    state_idxs = {
        "awake": np.array([]),
        "nrem": np.array([]),
        "rem": np.array([]),
        "transition": np.array([]),
    }

    def map_state(state: str, next_state: str | None) -> str:
        if state == "nrem" and next_state == "deep nrem":
            return "nrem"
        if state == "deep nrem" and next_state == "nrem":
            return "nrem"
        if state != next_state:
            return "transition"
        if state in {"movement", "awake"}:
            return "awake"
        if state in {"deep nrem", "nrem"}:
            return "nrem"
        if state == "rem":
            return "rem"
        raise ValueError(f"state {state} not recognized")

    for idx, state in enumerate(sleep_state):
        key = map_state(
            state, sleep_state[idx + 1] if idx + 1 < len(sleep_state) else None
        )

        state_idxs[key] = np.append(
            state_idxs[key],
            np.arange(idx * sampling_rate_lfp, (idx + 1) * sampling_rate_lfp),
        )

    included_idxs = np.sort(np.concatenate(list(state_idxs.values())))
    assert np.all(np.diff(included_idxs) == 1)
    assert (len(included_idxs) - n_samples) / sampling_rate_lfp < 1.5

    colors = ["blue", "green", "red"]

    if plot:
        plt.figure(figsize=(20, 4))

        for idx, state in enumerate(["rem", "awake", "nrem"]):
            plt.plot(
                state_idxs[state] / sampling_rate_lfp,
                np.ones_like(state_idxs[state]),
                ".",
                color=colors[idx],
            )
            plt.plot(
                seconds[sleep_state == state],
                np.ones_like(seconds[sleep_state == state]) + 1,
                ".",
                color=colors[idx],
                label=f"{state}",
            )

        plt.plot(
            state_idxs["transition"] / sampling_rate_lfp,
            np.ones_like(state_idxs["transition"]),
            ".",
            color="black",
        )

        plt.ylim(0, 4)
        plt.legend()

    return state_idxs
