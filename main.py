from pathlib import Path
import random
from typing import List, Tuple

import numpy as np

from data_import import Session


from models import LED, Sound

import matplotlib.pyplot as plt

from utils import extract_frames_fast, get_number_of_frames, save_video
from video_timestamps import get_creation_time, get_frame_timestamps
from ripples.utils_npyx import load_lfp_npyx, load_sync_npyx
from ripples.utils import threshold_detect

from rsync import Rsync_aligner


HERE = Path(__file__).parent


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

    if session.task_name == "sleeping":
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


def main(data_folder: Path, lfp_path: Path) -> None:

    video_files = sorted(list(data_folder.glob("*.mp4")))
    trigger_files = sorted(list(data_folder.glob("*time.npy")))
    pycontrol_files = sorted(list(data_folder.glob("*.tsv")))
    assert (
        len(video_files) == len(trigger_files) == len(pycontrol_files)
    ), "Mismatch in number of files, Got {len(video_files)} videos, {len(trigger_files)} triggers and {len(pycontrol_files)} pycontrol files"

    n_video_frames = [get_number_of_frames(video_file) for video_file in video_files]
    n_triggers = [
        np.load(trigger_file, allow_pickle=True).shape[0]
        for trigger_file in trigger_files
    ]
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    assert np.all(
        n_video_frames == n_triggers
    ), f"Number of video frames and triggers do not match, this may happen, look for off-by-ones. Got {n_video_frames} video frames, and {n_triggers} triggers"

    print("Startng lfp load")
    # raw_sync = load_sync_npyx(lfp_path)
    # np.save("RAW_SYNC.npy", raw_sync)

    raw_sync = np.load("RAW_SYNC.npy")
    print("finished lfpd  load")
    sync_npx = threshold_detect(raw_sync, 0.5)

    rsync_times = [session.times["rsync"] for session in sessions]
    assert sum(len(rs) for rs in rsync_times) == len(sync_npx)

    chunk_start = 0
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

    1 / 0

    # rsync = Rsync_aligner(
    #     sync_npx,
    #     session.times["rsync"],
    # )

    sounds, leds = process_session(session)
    n_frames = get_number_of_frames(video_path)
    camera_frame_times = np.load(frame_trigger_time_path, allow_pickle=True)

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

    ########################################## Second Session ########################################################
    # frame_trigger_time_path = Path(
    #     "/Volumes/MarcBusche/Qichen/Neuropixels/mice/ELGH-09266/behavior/pycontrol data/20250424/09266-2025-04-24-230818_frame_trigger.time.npy"
    # )

    # session_path = Path(
    #     "/Volumes/MarcBusche/Qichen/Neuropixels/mice/ELGH-09266/behavior/pycontrol data/20250424/09266-2025-04-24-222710.tsv"
    # )
    # video_path = Path(
    #     "/Volumes/MarcBusche/Qichen/Neuropixels/mice/ELGH-09266/behavior/video/09266_2025-04-24-230825-0000.mp4"
    # )
    # lfp_path = Path(
    #     "/Volumes/MarcBusche/Qichen/Neuropixels/mice/ELGH-09266/2050424_g0/2050424_g0_imec0"
    # )
    data_folder = Path("/Volumes/MarcBusche/James/Alex/2025-05-16/11151")
    lfp_path = data_folder / "20250516_g0" / "20250516_g0_imec0"
    main(data_folder, lfp_path)
