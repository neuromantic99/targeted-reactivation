from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matlab.engine

from data_import import Session


from lfp import plot_ripple_power_by_channel, plot_spectrogram, plot_swr_with_regions, plot_ripple_power_combined_processing
from nte import map_channels_to_regions
from models import LED, Sound
from video_adjuster import adjust_video_duration, get_video_duration, adjust_video_to_lfp


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

def plot_sync_alignment(aligner, session, video_frames, chunk_start=None, chunk_end=None, video_fps=30):

    plt.figure(figsize=(16, 10))
    
    # Convert units and get pulse times
    npx_pulses = aligner.pulse_times_A / 2500  # Neuropixel to seconds
    pc_pulses = session.times['rsync']         # PyControl pulses
    video_end = video_frames / video_fps       # Video duration
    
    # Main pulse plot (vertical lines)
    plt.vlines(npx_pulses, ymin=1, ymax=1.2, color='r', linewidth=0.5, label='Neuropixel')
    plt.vlines(pc_pulses, ymin=0.8, ymax=1, color='g', linewidth=0.5, label='PyControl')
    
    # Mark chunk boundaries (passed as parameters)
    if chunk_start is not None:
        plt.axvline(chunk_start, color='purple', linestyle='--', linewidth=2, label='Chunk start')
    if chunk_end is not None:
        plt.axvline(chunk_end, color='orange', linestyle='--', linewidth=2, label='Chunk end')
    
    # Highlight the actual last pyControl pulse
    last_pc = pc_pulses[-1]
    plt.axvline(last_pc, color='cyan', linestyle=':', linewidth=1.5, alpha=0.7, label='True pyControl end')
    
    # 50-second grid lines
    max_time = max(npx_pulses[-1], video_end) if len(npx_pulses) > 0 else video_end
    for t in np.arange(0, max_time + 50, 50):
        plt.axvline(t, color='gray', alpha=0.15, linestyle='-')
    
    # Video frames (thin blue ticks at bottom)
    frame_times = np.arange(video_frames) / video_fps
    plt.vlines(frame_times, ymin=0, ymax=0.2, color='b', linewidth=0.2, alpha=0.3, label='Video Frames')
    
    # Format main plot
    plt.yticks([0.1, 0.9, 1.1], ['Video', 'PyControl', 'Neuropixel'])
    plt.ylim(0, 1.3)
    plt.xticks(np.arange(0, max_time + 50, 50))
    plt.xlabel('Time (seconds)')
    
    # Calculate offset between chunk end and last pyControl pulse
    offset = chunk_end - last_pc if chunk_end is not None else 0
    
    # Create informative title
    title = (f"Sync Pulse Alignment\n"
             f"Video: {video_end:.1f}s | "
             f"NPX: {npx_pulses[-1]:.1f}s | "
             f"Offset: {offset:.3f}s")
    plt.title(title, pad=20)
    
    # Add detailed statistics box
    stats_text = (f"Alignment Diagnostics:\n"
                  f"Chunk start: {chunk_start:.3f}s\n"
                  f"Chunk end: {chunk_end:.3f}s\n"
                  f"True pyControl end: {last_pc:.3f}s\n"
                  f"Neuropixel pulses: {len(npx_pulses)}\n"
                  f"pyControl pulses: {len(pc_pulses)}")
    
    plt.gca().text(0.02, 0.95, stats_text, 
                  transform=plt.gca().transAxes,
                  verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.8))
    
    # Add offset warning if significant
    if abs(offset) > 0.1:  # More than 100ms offset
        warning_text = (f"WARNING: Significant offset!\n"
                       f"Chunk end is {offset:.3f}s\n"
                       f"{'after' if offset > 0 else 'before'} last pyControl pulse")
        
        plt.gca().text(0.5, 0.95, warning_text,
                      transform=plt.gca().transAxes,
                      verticalalignment='top',
                      horizontalalignment='center',
                      bbox=dict(facecolor='red', alpha=0.3))
    
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
    
    # 60-80 second zoomed inset
    ax_inset = plt.axes([0.5, 0.5, 0.35, 0.35])
    zoom_window = (60, 80)
    
    # Filter pulses to zoom window
    npx_in_zoom = npx_pulses[(npx_pulses >= zoom_window[0]) & (npx_pulses <= zoom_window[1])]
    pc_in_zoom = pc_pulses[(pc_pulses >= zoom_window[0]) & (pc_pulses <= zoom_window[1])]
    
    ax_inset.vlines(npx_in_zoom, 1, 1.2, color='r', linewidth=1)
    ax_inset.vlines(pc_in_zoom, 0.8, 1, color='g', linewidth=1)
    
    # Add chunk markers to zoom if visible
    if chunk_start and zoom_window[0] <= chunk_start <= zoom_window[1]:
        ax_inset.axvline(chunk_start, color='purple', linestyle='--', linewidth=1.5)
    if chunk_end and zoom_window[0] <= chunk_end <= zoom_window[1]:
        ax_inset.axvline(chunk_end, color='orange', linestyle='--', linewidth=1.5)
    
    # Format inset
    ax_inset.set_xlim(zoom_window)
    ax_inset.set_xticks(np.arange(60, 81, 5))
    ax_inset.set_yticks([])
    ax_inset.set_title('Zoom: 60-80 seconds', pad=10)
    ax_inset.grid(alpha=0.2)
    
    plt.tight_layout()
    return plt.gcf()

def plot_sync_vs_camera_times(aligner):
    # Convert both timebases to the same unit (ms)
    pulse_times_A_ms = aligner.pulse_times_A * aligner.units_A
    pulse_times_B_ms = aligner.pulse_times_B * aligner.units_B

    # Plot B vs A (i.e., ephys/camera vs pyControl)
    plt.figure(figsize=(6, 6))
    plt.plot(pulse_times_A_ms, pulse_times_B_ms, '.', markersize=3)
    plt.xlabel("pyControl pulse times (ms)")
    plt.ylabel("ephys/camera pulse times (ms)")
    plt.title("Before alignment: pulse time comparison in common units")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def check_recording_durations(data_folder, lfp_path, video_frames, session, aligner, session_idx, total_sessions):

    # Load raw sync data
    mouse = data_folder.name
    raw_sync_path = Path(f"raw_sync_{mouse}.npy")
    if raw_sync_path.exists():
        raw_sync = np.load(raw_sync_path)
    else:
        raw_sync = load_sync_npyx(lfp_path)
        np.save(raw_sync_path, raw_sync)

    # Get sync pulses (ensure integer indices)
    sync_npx = threshold_detect(raw_sync, 0.5).astype(int)  # Convert to integers
    
    # Convert to integer indices
    session_start = int(aligner.first_matched_time_A)
    session_end = int(aligner.last_matched_time_A)
    
    # Calculate durations
    lfp_duration = (session_end - session_start) / 2500  # seconds
    
    # Find pulses in this session (using integer comparison)
    session_sync_mask = (sync_npx >= session_start) & (sync_npx <= session_end)
    session_sync_pulses = sync_npx[session_sync_mask]
    
    pulse_duration = 0
    if len(session_sync_pulses) > 1:
        pulse_duration = (session_sync_pulses[-1] - session_sync_pulses[0]) / 2500

    # Other durations
    video_duration = video_frames / 30
    pc_pulses = session.times['rsync']
    pc_duration = pc_pulses[-1] - pc_pulses[0] if len(pc_pulses) > 1 else 0
    aligned_duration = (session_end - session_start) / 2500

    # Create results table
    durations = {
        "Session": f"{session_idx+1}/{total_sessions}",
        "Raw LFP (s)": lfp_duration,
        "Video (s)": video_duration,
        "NPX Sync (s)": pulse_duration,
        "pyControl (s)": pc_duration,
        "Aligned (s)": aligned_duration
    }

    # Print results
    print("\n=== Session Duration Comparison ===")
    for name, dur in durations.items():
        if isinstance(dur, str):
            print(f"{name}: {dur}")
        else:
            print(f"{name}: {dur:.3f}")
    print("Start:", session.times['run_start'])
    print("End:", session.times['run_end'])

    # Plot with proper integer indexing
    plt.figure(figsize=(15, 5))
    
    # Ensure we don't exceed array bounds
    safe_end = min(session_end, len(raw_sync)-1)
    lfp_segment = raw_sync[session_start:safe_end+1]
    
    # Time axis in seconds
    time_axis = np.arange(session_start, safe_end+1)/2500
    
    plt.plot(time_axis, lfp_segment, alpha=0.5)
    
    # Only plot pulses within bounds
    valid_pulses = session_sync_pulses[(session_sync_pulses >= session_start) & 
                                      (session_sync_pulses <= safe_end)]
    if len(valid_pulses) > 0:
        plt.scatter(valid_pulses/2500, 
                   np.ones(len(valid_pulses))*np.max(lfp_segment)*1.1,
                   color='r', s=10)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    plt.title(f'Session {session_idx+1} Sync ({lfp_duration:.1f}s)')
    plt.tight_layout()
    
    return plt.gcf(), durations

def main(data_folder: Path, lfp_path: Path) -> None:
    mouse = data_folder.name
    video_files, trigger_files, pycontrol_files = get_data_paths(data_folder)

    n_video_frames = [get_number_of_frames(video_file) for video_file in video_files]

    n_triggers = [
        np.load(trigger_file, allow_pickle=True).shape[0]
        for trigger_file in trigger_files
    ]

    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]
    adjusted_dir = data_folder / "time_adjusted_videos"
    adjusted_dir.mkdir(exist_ok=True, parents=True)
    for i, session in enumerate(sessions):
        print(f"Session {i}: run_start={session.run_start}, run_end={session.run_end}")
        print(f"Session {i} times keys: {session.times.keys()}")  # Also check what's in times

    task_names = [session.task_name.replace(" ", "_").lower() for session in sessions]
    extended_pc_times = []

    for session in sessions:
        run_start = session.times['run_start'][0]  # Access first element of array
        rsync_pulses = session.times['rsync']
        run_end = session.times['run_end'][0]
        extended_pc_times.extend([run_start, *rsync_pulses, run_end])
        print(f"Extended times for session: start={run_start}, end={run_end}, #rsync={len(rsync_pulses)}")

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

    chunk_start = 0
    # A list of Rsync_aligner objects, one for each session
    # So in theory, aligner 0 is the conditioning aligner.
    aligners = []
    for i, rsync_time in enumerate(rsync_times):
        aligner = Rsync_aligner(
            sync_npx[chunk_start : chunk_start + len(rsync_time)],
            rsync_time,
            raise_exception=True,
        )
        aligners.append(aligner)
        chunk_start += len(rsync_time)
    
    
    # Get the extended pyControl times (including run_start/end)
    extended_pc_times = []
    for session in sessions:
        # Safely get all required times
        run_start = session.times.get('run_start', np.array([0]))[0]
        rsync_pulses = session.times.get('rsync', np.array([]))
        run_end = session.times.get('run_end', np.array([0]))[0]
        
        extended_pc_times.extend([run_start, *rsync_pulses, run_end])

    # Convert these extended pyControl times to neuropixel times
    extended_npx_times = []
    chunk_start = 0
    for aligner, rsync_time in zip(aligners, rsync_times):
        # Convert each session's extended times
        session_pc_times = np.array(extended_pc_times[chunk_start : chunk_start + len(rsync_time) + 2])
        print(f"Converting PC times to NPX: {session_pc_times}")
        session_npx_times = aligner.B_to_A(session_pc_times, extrapolate=True)
        print(f"Resulting NPX times: {session_npx_times}")
        extended_npx_times.extend(session_npx_times)
        chunk_start += len(rsync_time) + 2

    # Replace sync_npx with the extended version
    sync_npx = np.array(extended_npx_times)
    print(f"Total rsync pulses from pycontrol: {sum(len(rs) for rs in rsync_times)}")
    print(f"Number of sync_npx pulses after extension: {len(sync_npx)}")
    print(f"Number of sessions: {len(rsync_times)}")
    print(f"Expected extended length: {sum(len(rs) + 2 for rs in rsync_times)}")
    expected_length = sum(len(rs) + 2 for rs in rsync_times)
    assert expected_length == len(sync_npx), f"Expected {expected_length} sync pulses after extension, got {len(sync_npx)}"

    ADJUSTMENT_THRESHOLD = 0.1  # Only adjust if difference >100ms
    MIN_DURATION = 1.0          # Minimum valid video duration (seconds)
    
    for i, (video_file, session) in enumerate(zip(video_files, sessions)):
        print(f"\nSession {i}: {video_file.name}")
        
        try:
            # Get durations
            video_duration = get_video_duration(video_file)
            lfp_duration = session.run_end - session.run_start
            duration_diff = lfp_duration - video_duration
            abs_diff = abs(duration_diff)
            
            print(f"  Video: {video_duration:.3f}s | LFP: {lfp_duration:.3f}s")
            print(f"  Difference: {duration_diff:+.3f}s ({duration_diff/video_duration*100:+.2f}%)")
            
            # Validate durations
            if video_duration < MIN_DURATION:
                raise ValueError(f"Video too short ({video_duration:.1f}s)")
            if lfp_duration < MIN_DURATION:
                raise ValueError(f"LFP duration too short ({lfp_duration:.1f}s)")
            
            # Only adjust if difference exceeds threshold
            if abs_diff <= ADJUSTMENT_THRESHOLD:
                print("  Difference within tolerance, using original video")
                continue
                
            output_path = adjusted_dir / f"adj_{duration_diff:+.1f}s_{video_file.name}"
            
            print(f"  Adjusting {'longer' if duration_diff > 0 else 'shorter'} by {abs_diff:.3f}s")
            if adjust_video_to_lfp(video_file, lfp_duration, output_path):
                video_files[i] = output_path
                print("  Adjustment successful")
            else:
                print("  Using original due to adjustment failure")
                
        except Exception as e:
            print(f"  Error processing video: {str(e)}")
            print("  Using original video as fallback")
    
    area_channel = map_channels_to_regions(-2500,-500,270,60,3200,384)
    total_sessions = len(sessions)
    sessions_list = []
    for i, (aligner, session) in enumerate(zip(aligners, sessions)):
        print(f"\nProcessing session {i+1}/{len(aligners)}")
        aligner_id = f"{mouse}_{task_names[i]}_{i}"
        sessions_list.append((lfp_path, session, (88, 133), (180, 320), aligner_id)) 
        print(f"\nUnits A {aligner.units_A}, Units B {aligner.units_B}")
        print(f"\nStart A {aligner.first_matched_time_A}, Start B {aligner.first_matched_time_B}")
        print(f"\nEnd A {aligner.last_matched_time_A}, End B {aligner.last_matched_time_B}")
        
        
        plot_ripple_power_by_channel(
            lfp_path=lfp_path,
            mouse=mouse,
            aligner=aligner,
            session=session,
            ca1_range=(88, 133),
            rsc_range=(180, 320),
            aligner_id=aligner_id,
            area_channel=area_channel  # Pass pre-computed mapping
        )

        
        fig = plot_sync_alignment(
        aligner, 
        session, 
        n_video_frames[i],
        #chunk_start=aligner.first_matched_time_A / aligner.units_B,
        #chunk_end=aligner.last_matched_time_A / aligner.units_B,
        chunk_start=run_start,
        chunk_end=run_end,
        )
        fig.savefig(f"sync_alignment_{mouse}_session{i}.png", dpi=300)
        plt.close(fig)
        last_pc = sessions[i].times['rsync'][-1]
    last_match = aligner.last_matched_time_A / 2500
    print(f"Session {i}: Last pyControl={last_pc:.3f}s, Last match={last_match:.3f}s, Offset={last_match-last_pc:.3f}s")
    

    print("finished channel mapping")
    # Print each channel with its corresponding brain region
    for channel_num, region in enumerate(area_channel):
        print(f"Channel {channel_num}: {region}")

    for i, session in enumerate(sessions):
        print(f"Session {i}: {session.task_name} (duration: {session.times['rsync'][-1] - session.times['rsync'][0]} seconds)")

    sessions_list = []
    for i, (aligner, session) in enumerate(zip(aligners, sessions)):
        aligner_id = f"{mouse}_{task_names[i]}_{i}"
        sessions_list.append((
            lfp_path,  # Path isn't actually used when cache exists
            session,
            (88, 133), 
            (180, 320),
            aligner_id
        ))

    # Suraya's request to do the filters on the combined data for all 3 sessions
    plot_ripple_power_combined_processing(sessions_list)
  


    
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
    data_folder = Path("Z:/Alex/Reactivations/2025-05-19/11151")
    lfp_path = data_folder / "20250519_g0" / "20250519_g0_imec0"
    main(data_folder, lfp_path)
    #############################################

    ################# 11150 ######################

    # data_folder = Path("/Volumes/MarcBusche/Alex/Reactivations/2025-05-20/11150")
    # lfp_path = data_folder / "20250520_g0" / "20250520_g0_imec0"
    # main(data_folder, lfp_path)