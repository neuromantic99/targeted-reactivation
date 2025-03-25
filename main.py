from pathlib import Path
from typing import List, Tuple

import numpy as np

from data_import import Session
from models import LED, Sound

import matplotlib.pyplot as plt

from utils import extract_frames_fast, save_video

HERE = Path(__file__).parent


def process_session(session: Session) -> Tuple[List[Sound], List[LED]]:

    sound_prints = [
        printed
        for printed in session.prints
        if printed.string.startswith("Deliverying sound frequency")
    ]
    sounds = [
        Sound(sound.time, int(sound.string.split("Deliverying sound frequency ")[1]))
        for sound in sound_prints
    ]

    led_prints = [
        printed
        for printed in session.prints
        if printed.string.startswith("Turning on LED Color")
    ]

    leds = [
        LED(led.time, led.string.split("Turning on LED Color: ")[1])
        for led in led_prints
    ]
    # TODO: This is the wrong way round
    for sound, led in zip(sounds, leds, strict=True):
        assert 1.99 < led.time - sound.time < 2.01

    return sounds, leds


def main() -> None:

    sync_umbrella = Path(
        "/Volumes/MarcBusche/Qichen/Neuropixels/reactivation/behavior/"
    )

    video_umbrella = Path(
        "/Volumes/MarcBusche/Qichen/Neuropixels/reactivation/Videos/20250321"
    )

    sounds, leds = process_session(
        Session(sync_umbrella / "09266-2025-03-21-165858.tsv")
    )

    camera_frame_times = np.load(
        sync_umbrella / "09266-2025-03-21-165858_frame_trigger.time.npy"
    )

    video_path = video_umbrella / "09266_2025-03-21-165852-0000.mp4"

    # n_frames = process_video_ffmpeg(video_path, chunk_size=10)
    n_frames = 108000

    # Assumes camera was running when pycontrol started
    camera_frame_times = camera_frame_times[:n_frames]

    # get_sound_videos(sounds, camera_frame_times, video_path)
    get_light_videos(leds, camera_frame_times, video_path)


def get_sound_videos(
    sounds: List[Sound], camera_frame_times: np.ndarray, video_path: Path
) -> None:
    for idx, sound in enumerate(sounds):
        if idx > 50:
            break
        trial_start = sound.time - 1
        trial_end = sound.time + 1
        frames_trial = np.where(
            np.logical_and(
                camera_frame_times >= trial_start, camera_frame_times <= trial_end
            )
        )[0]
        trial = extract_frames_fast(video_path, frames_trial)[:, :, :, 0]
        trial = np.swapaxes(np.swapaxes(trial, 0, 2), 1, 2)
        save_video(trial, output_path=str(HERE / "videos" / f"trial_{idx}_sound.mp4"))

        print(f"Done for trial {idx}")


def get_light_videos(
    leds: List[LED], camera_frame_times: np.ndarray, video_path: Path
) -> None:
    for idx, led in enumerate(leds):
        if idx > 50:
            break
        trial_start = led.time - 1
        trial_end = led.time + 1
        frames_trial = np.where(
            np.logical_and(
                camera_frame_times >= trial_start, camera_frame_times <= trial_end
            )
        )[0]
        trial = extract_frames_fast(video_path, frames_trial)[:, :, :, 0]
        trial = np.swapaxes(np.swapaxes(trial, 0, 2), 1, 2)
        save_video(trial, output_path=str(HERE / "videos" / f"trial_{idx}_light.mp4"))

        print(f"Done for trial {idx}")


if __name__ == "__main__":
    main()
