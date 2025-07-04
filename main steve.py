from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from data_import import Session


from lfp import plot_lfps, plot_spectrogram
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

    print(f"\nSession Debug: {session.file_name}")
    print(f"\nSession Debug: {session.task_name}")
    print(f"Total sounds: {len(sounds)} | Total LEDs: {len(leds)}")
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


def load_data(
    data_folder: Path,
) -> Tuple[List[Path], List[Path], List[Path]]:
    return (
        sorted(list(data_folder.glob("*.mp4"))),
        sorted(list(data_folder.glob("*time.npy"))),
        sorted(list(data_folder.glob("*.tsv"))),
    )


def main_11153(data_folder: Path, lfp_paths: List[Path]) -> None:
    mouse = data_folder.name

    video_files, trigger_files, pycontrol_files = load_data(data_folder)
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

    plot_lfps(lfp_paths, mouse, aligners)


#def main(data_folder: Path, lfp_path: Path) -> None:
def main(data_folder: Path, lfp_path: Path) -> Tuple[List[Session], List[Path], List[Path]]:
    mouse = data_folder.name
    video_files, trigger_files, pycontrol_files = load_data(data_folder)

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

    sync_npx = threshold_detect(raw_sync, 0.5)
    rsync_times = [session.times["rsync"] for session in sessions]
    assert sum(len(rs) for rs in rsync_times) == len(sync_npx)

    chunk_start = 0

    chunk_start = np.sum([len(times) for times in rsync_times[:6]])
    sync_npx = sync_npx[chunk_start:]
    aligners = []
    for rsync_time in rsync_times[6:]:
        aligners.append(
            Rsync_aligner(
                sync_npx[chunk_start : chunk_start + len(rsync_time)],
                rsync_time,
                raise_exception=True,
            )
        )

        chunk_start += len(rsync_time)

 #   1 / 0
    
   # sounds, leds = process_session(session)
   # n_frames = get_number_of_frames(video_path)
   # camera_frame_times = np.load(frame_trigger_time_path, allow_pickle=True)
    
    if lfp_path.exists():
        from lfp import plot_lfps, plot_spectrogram
        
        print(f"\nProcessing LFP data from: {lfp_path}")
        
        try:
            # SAFE ACCESS TO ALIGNERS
            if aligners:  # Only plot if we have aligners
                plot_lfps([lfp_path], mouse, aligners)
            else:
                print("Warning: No aligners available - skipping time-aligned LFP plots")
            
            # BASIC SPECTROGRAM (doesn't need aligners)
            try:
                lfp_data = load_lfp_reactivations(
                    lfp_path,
                    chunk_start_seconds=0,
                    chunk_end_seconds=30  # First 30 seconds
                )
                if lfp_data is not None:
                    plot_spectrogram(lfp_data[:300], 2500)  # Only first 300 channels
                else:
                    print("Warning: No LFP data loaded")
            except Exception as e:
                print(f"Spectrogram failed: {str(e)}")
                
        except Exception as e:
            print(f"LFP plotting failed: {str(e)}")
    else:
        print(f"Warning: LFP path not found at {lfp_path}")

    
    return sessions, video_files, trigger_files

    # get_sound_videos(sounds, camera_frame_times, video_path)
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
        "Z:/Suraya/NPX/Sleep/Raw/01040_7M_S305N_EC_20241102/20241102_g1/20241102_g1_imec0"
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

    ################ 11151 ######################
    data_folder = Path("Z:/Alex/Reactivations/2025-05-19/11151")
    lfp_path = data_folder / "20250519_g0" / "20250519_g0_imec0"
   # main(data_folder, lfp_path)
    sessions, video_files, trigger_files = main(data_folder, lfp_path)
    for i, session in enumerate(sessions[6:]):  # Skip first 6 sessions
        if session.task_name != "conditioning_alex":
            print(f"Skipping session {i+6} (task: {session.task_name})")
            continue
        sounds, leds = process_session(session)
        video_path = video_files[i+6]  # Match session index to video
        frame_trigger_time_path = trigger_files[i+6]
        camera_frame_times = np.load(frame_trigger_time_path, allow_pickle=True)
        get_light_videos(leds, camera_frame_times, video_path)


import pandas as pd
import matplotlib.pyplot as plt

# Create session summary dataframe
session_data = []
for i, session in enumerate(sessions):
    if session.task_name != "conditioning_alex":
            print(f"Skipping session {i+6} (task: {session.task_name})")
            continue
    sounds, leds = process_session(session)
    session_data.append({
        'Session': i,
        'Start Time': session.datetime,
        'Duration (min)': (session.events[-1].time - session.events[0].time)/60,
        'Sounds': len(sounds),
        'LEDs': len(leds) if leds else 0,
        'Video File': video_files[i].name if i < len(video_files) else None,
        'Trigger File': trigger_files[i].name if i < len(trigger_files) else None
    })

df = pd.DataFrame(session_data)
from IPython.display import display
display(df)
# First collect all stimulus events
all_events = []
for i, session in enumerate(sessions):
    if session.task_name != "conditioning_alex":
            print(f"Skipping session {i+6} (task: {session.task_name})")
            continue
    sounds, leds = process_session(session)
    
    # Add sound events
    for sound in sounds:
        all_events.append({
            'Session': i,
            'Type': 'Sound',
            'Frequency (Hz)': sound.frequency, 
            'Time (s)': sound.time,
            'Video File': video_files[i].name if i < len(video_files) else None
        })
    
    # Add LED events
    if leds:
        for led in leds:
            all_events.append({
                'Session': i,
                'Type': 'LED',
                'Color': led.color, 
                'Time (s)': led.time,
                'Video File': video_files[i].name if i < len(video_files) else None
            })

# Create DataFrame and export
events_df = pd.DataFrame(all_events)
events_excel_path = HERE / "stimulus_events.xlsx"
events_df.to_excel(events_excel_path, index=False)
print(f"Stimulus events saved to: {events_excel_path}")


