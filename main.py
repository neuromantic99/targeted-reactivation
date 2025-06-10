from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data_import import Session


from lfp import plot_ripple_power_by_channel, plot_spectrogram
from models import LED, Sound


from utils import extract_frames_fast, get_number_of_frames, save_video
from ripples.utils_npyx import load_sync_npyx, load_lfp_reactivations
from ripples.utils import threshold_detect


from ripples.utils import (
    bandpass_filter,
    compute_power,
    compute_envelope,
)

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


def get_data_paths(
    data_folder: Path,
) -> Tuple[List[Path], List[Path], List[Path]]:
    return (
        sorted(list(data_folder.glob("*.mp4"))),
        sorted(list(data_folder.glob("*time.npy"))),
        sorted(list(data_folder.glob("*.tsv"))),
    )


def main_11153(data_folder: Path, lfp_paths: List[Path]) -> None:
    mouse = data_folder.name

    video_files, trigger_files, pycontrol_files = get_data_paths(data_folder)
    n_video_frames = [get_number_of_frames(video_file) for video_file in video_files]
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    n_triggers = [
        np.load(trigger_file, allow_pickle=True).shape[0]
        for trigger_file in trigger_files
    ]

    raw_sync = []
    for idx, lfp_path in enumerate(lfp_paths):

        raw_sync_path = Path(f"raw_sync_{idx}_{mouse}.npy")
        if raw_sync_path.exists():
            raw_sync.append(np.load(f"raw_sync_{idx}_{mouse}.npy"))
        else:
            raw_sync.append(load_sync_npyx(lfp_path))
            np.save(raw_sync_path, raw_sync[idx])

    # Ignore the second recording for now
    n_video_frames = [n_video_frames[2], n_video_frames[3], n_video_frames[4]]
    n_triggers = [n_triggers[3], n_triggers[4], n_triggers[5]]
    sessions = [sessions[3], sessions[4], sessions[5]]

    sync_npx = threshold_detect(raw_sync[0], 0.5)
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

    plot_ripple_power_by_channel(lfp_paths, mouse, aligners)


def main(data_folder: Path, lfp_path: Path) -> None:
    mouse = data_folder.name
    video_files, trigger_files, pycontrol_files = get_data_paths(data_folder)

    n_video_frames = [get_number_of_frames(video_file) for video_file in video_files]

    n_triggers = [
        np.load(trigger_file, allow_pickle=True).shape[0]
        for trigger_file in trigger_files
    ]

    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    assert (
        len(n_video_frames) == len(n_triggers) == len(sessions)
    ), f"Mismatch in number of files, Got {len(n_video_frames)} videos, {len(n_triggers)} triggers and {len(sessions)} pycontrol files"

    if n_video_frames != n_triggers:
        print(
            f"Number of video frames and triggers do not match, this may happen. Got {n_video_frames} video frames, and {n_triggers} triggers"
        )

    print("Starting lfp load")
    raw_sync_path = Path(f"raw_sync_{mouse}.npy")
    if raw_sync_path.exists():
        raw_sync = np.load(raw_sync_path)
    else:
        raw_sync = load_sync_npyx(lfp_path)
        np.save(raw_sync_path, raw_sync)

    # The times of the sync pulse recorded on the NPX
    sync_npx = threshold_detect(raw_sync, 0.5)

    # The time of the sync pulse recorded on pycontrol
    rsync_times = [session.times["rsync"] for session in sessions]

    assert sum(len(rs) for rs in rsync_times) == len(sync_npx)

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
    plot_ripple_power_by_channel(lfp_path=lfp_path, mouse=mouse, aligner=aligners[0])

    # sounds, leds = process_session(session)
    # n_frames = get_number_of_frames(video_path)
    # camera_frame_times = np.load(frame_trigger_time_path, allow_pickle=True)

    # # get_sound_videos(sounds, camera_frame_times, video_path)
    # get_light_videos(leds, camera_frame_times, video_path)


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


def test_suraya_data() -> None:
    data_folder = Path(
        "/Volumes/MarcBusche/Suraya/NPX/Sleep/Raw/01040_7M_S305N_EC_20241102/20241102_g1/20241102_g1_imec0"
    )

    lfp = np.load("lfp_suraya.npy")
    # lfp = load_lfp_reactivations(
    #     data_folder, chunk_start_seconds=0, chunk_end_seconds=5 * 60
    # )

    # np.save("lfp_suraya.npy", lfp)

    RIPPLE_BAND = [120, 250]
    # lfp = lfp[:300, :]
    lfp = lfp - np.mean(lfp, axis=1, keepdims=True)
    plot_spectrogram(lfp, 2500)

    # Get the channels with the max SWR channel
    # First 5 min of the recording
    swr_power = compute_power(
        bandpass_filter(
            lfp[:, : 2500 * 5 * 60], RIPPLE_BAND[0], RIPPLE_BAND[1], 2500, order=4
        )
    )
    plt.plot(compute_power(lfp[:, : 2500 * 5 * 60]))
    1 / 0


if __name__ == "__main__":

    ################# 11153 ######################
    # data_folder = Path("/Volumes/MarcBusche/Alex/Reactivations/2025-05-18/11153")
    # lfp_paths = [
    #     data_folder / "20250518_g0" / "20250518_g0_imec0",
    #     data_folder / "20250518_g1" / "20250518_g1_imec0",
    # ]
    # main_11153(data_folder, lfp_paths)
    #############################################

    ################# 11151 ######################
    data_folder = Path("/Volumes/MarcBusche/Alex/Reactivations/2025-05-19/11151")
    lfp_path = data_folder / "20250519_g0" / "20250519_g0_imec0"
    main(data_folder, lfp_path)
    #############################################

    ################# 11150 ######################

    # data_folder = Path("/Volumes/MarcBusche/Alex/Reactivations/2025-05-20/11150")
    # lfp_path = data_folder / "20250520_g0" / "20250520_g0_imec0"
    # main(data_folder, lfp_path)
