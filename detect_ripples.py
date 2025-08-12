import json
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from ripples.utils import compute_power, bandpass_filter, threshold_detect
import seaborn as sns
import traceback


from npyx import extract_rawChunk, read_metadata

from ripples.utils_npyx import load_sync_npyx
from ripples.ripple_detection import (
    get_candidate_ripples,
    remove_duplicate_ripples,
    get_quality_metrics,
)
from consts import DETECTION_METHOD, LOCAL_SSD, RIPPLE_BAND
from gsheets_importer import gsheet2df
from models import RipplesCache
from plotting import plot_lfp_profile, plot_lfp_spectrogram

from data_import import Session
from main import HERE, get_aligners

import matplotlib.pyplot as plt

from utils import get_data_paths

HERE = Path(__file__).parent


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


def get_detection_channels_CA1(
    ca1: np.ndarray, ca1_low: int, ca1_high: int, sampling_rate_lfp: float
) -> np.ndarray:

    # # Find CA1 channel with highest Ripple power and +/- two channels to detect ripples, then do CAR
    swr_power = compute_power(
        bandpass_filter(ca1, RIPPLE_BAND[0], RIPPLE_BAND[1], sampling_rate_lfp, order=4)
    )

    ca1_max_power_channel = np.nanargmax(swr_power)

    if ca1_max_power_channel - 2 < 0:
        ca1_max_power_channel += 1
        print("Detection channels moved to stay in CA1")

    all_ca1_channels = np.arange(ca1_low, ca1_high)
    assert ca1_max_power_channel + 3 < len(all_ca1_channels)
    assert ca1_max_power_channel - 2 >= 0

    # CA1_channels are the channels in CA1 used for ripple detection
    detection_channels_ca1 = all_ca1_channels[
        ca1_max_power_channel - 2 : ca1_max_power_channel + 3
    ]

    # If the reference channel is part of the selected channels for ripple analysis replace with neighbouring channel with the higher ripple power
    if 191 in detection_channels_ca1:
        detection_channels_ca1.remove(191)
        lower_channel = all_ca1_channels[ca1_max_power_channel - 3]
        if (ca1_max_power_channel + 3) > (len(all_ca1_channels) - 1):
            swr_pow_higher_channel = 0
        else:
            higher_channel = all_ca1_channels[ca1_max_power_channel + 3]
            swr_pow_higher_channel = swr_power[higher_channel]
        if swr_power[lower_channel] > swr_pow_higher_channel:
            detection_channels_ca1.append(lower_channel)
        else:
            detection_channels_ca1.append(higher_channel)

    return detection_channels_ca1


def get_ripple_rate(lfp_path: Path, region_channels: Tuple[int, int, int, int]) -> None:
    mouse = lfp_path.parent.parent.name
    imec = f"imec_{str(lfp_path).split('imec')[1]}"

    # if (HERE / "results" / "ripples" / f"{mouse}_{imec}.json").exists():
    #     print(f"Ripple results for {mouse}_{imec} already exist, skipping.")
    #     return

    ca1_low, ca1_high, rsc_low, rsc_high = region_channels
    data_folder = lfp_path.parent.parent

    _, frame_trigger_times, pycontrol_files = get_data_paths(data_folder)
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    assert len(sessions) == 3
    assert len(frame_trigger_times) == 3

    rsync_times = [session.times["rsync"] for session in sessions]

    meta = read_metadata(lfp_path)

    npx_sync_times = get_sync(lfp_path, mouse, imec)

    aligners = get_aligners(
        npx_sync_times,
        rsync_times,
    )

    # This should be very close to the sampling rate reported by neuropixels but this is slightly more
    # accurate when aligning to pycontrol. In practice either are probably fine.
    assert abs(aligners[1].units_B - meta["lowpass"]["sampling_rate"]) < 1
    sampling_rate_lfp = meta["lowpass"]["sampling_rate"]

    # start_rest2 =

    resting_lfp_path = LOCAL_SSD / f"resting_lfps/{mouse}_{imec}.npy"
    if (resting_lfp_path).exists():
        print("loading existing resting LFP")
        lfp = np.load(resting_lfp_path)
    else:
        # start_rest = aligners[1].first_matched_time_A / sampling_rate_lfp
        # end_rest = aligners[1].last_matched_time_A / sampling_rate_lfp
        frame_triggers = np.load(frame_trigger_times[1])
        start_rest, end_rest = (
            aligners[1].B_to_A(np.array([frame_triggers[0], frame_triggers[-1]]))
            / sampling_rate_lfp
        )
        lfp = extract_rawChunk(
            lfp_path,
            [start_rest, end_rest],  # now taking the recording length as a float
            channels=np.arange(384),
            filt_key="lowpass",  # NPX data is devided in "high-pass" = spiking data and "low-pass" = LFP, no filter is being applied
            save=0,
            whiten=0,
            med_sub=False,
            hpfilt=False,
            hpfiltf=0,
            filter_forward=False,
            filter_backward=False,
            nRangeWhiten=None,
            nRangeMedSub=None,
            use_ks_w_matrix=True,
            ignore_ks_chanfilt=True,
            center_chans_on_0=False,
            verbose=True,
            scale=False,
            again=False,
        )

        np.save(resting_lfp_path, lfp)

    # plot_lfp_profile(
    #     lfp,
    #     ripple_band=RIPPLE_BAND,
    #     sampling_rate_lfp=sampling_rate_lfp,
    #     mouse=mouse,
    #     imec=imec,
    # )

    detect_ripples(mouse, imec, ca1_low, ca1_high, data_folder, sampling_rate_lfp, lfp)


def detect_ripples(
    mouse: str,
    imec: str,
    ca1_low: int,
    ca1_high: int,
    data_folder: Path,
    sampling_rate_lfp: float,
    lfp: np.ndarray,
) -> None:
    ca1_lfp = lfp[ca1_low:ca1_high, :]
    detection_channels_ca1 = get_detection_channels_CA1(
        ca1_lfp, ca1_low, ca1_high, sampling_rate_lfp
    )

    assert len(detection_channels_ca1) == 5, "Expected 5 detection channels in CA1"
    assert min(detection_channels_ca1) >= ca1_low
    assert max(detection_channels_ca1) < ca1_high

    ca1_lfp_detection_channels_only = lfp[detection_channels_ca1, :]
    common_average = np.nanmedian(ca1_lfp, axis=0)
    common_average_referenced = np.subtract(
        ca1_lfp_detection_channels_only, common_average
    )
    assert common_average_referenced.shape[0] == len(detection_channels_ca1)

    # Overwrite logic in Jana's code
    resting_ind = np.repeat(True, common_average_referenced.shape[1])

    candidate_events = get_candidate_ripples(
        common_average_referenced,
        detection_channels_ca1,
        resting_ind,
        sampling_rate_lfp,
        DETECTION_METHOD,
    )

    # Flattening makes further processing easier
    ripples = [event for events in candidate_events for event in events]

    print(f"Number of candidate ripples: {len(ripples)}")

    ripples = remove_duplicate_ripples(
        ripples, 0.05, sampling_rate_lfp
    )  # James 0.3, Buzaki 0.12, elife 0.05

    print(f"Number of ripples after removing duplicates: {len(ripples)}")

    freq_check, CAR_check, SRP_check, CAR_check_lr, SRP_check_lr, ripples = (
        get_quality_metrics(
            ripples, common_average_referenced, common_average, sampling_rate_lfp
        )
    )

    lfp_state_idx = get_lfp_index_sleep_state(
        data_folder=data_folder,
        # This mouse was done with buffering rather than streaming.
        # Need to remove ripples past the 30 minute mark below
        n_samples=lfp.shape[1] if mouse != "00053" else 30 * 60 * sampling_rate_lfp,
        sampling_rate_lfp=sampling_rate_lfp,
    )

    ripples = [
        ripple for ripple in ripples if ripple.peak_idx < sampling_rate_lfp * 30 * 60
    ]

    ripple_state = []

    for ripple in ripples:
        for state, idxs in lfp_state_idx.items():
            if ripple.peak_idx in idxs:
                ripple_state.append(state)
                break

    assert len(ripple_state) == len(ripples)

    total_passing_all = np.sum(
        np.array(freq_check) & np.array(CAR_check_lr) & np.array(SRP_check_lr)
    )

    cache_result = RipplesCache(
        candidate_events=ripples,
        common_average_reference_check=CAR_check_lr,
        frequency_check=freq_check,
        super_ripple_check=SRP_check_lr,
        length_recording=(
            lfp.shape[1] / sampling_rate_lfp if mouse != "00053" else 30 * 60
        ),
        state=ripple_state,
        state_lengths={state: len(list(idxs)) for state, idxs in lfp_state_idx.items()},
    )

    print(f"Total ripples passing all quality metrics: {total_passing_all}")
    print(f"Ripple rate {total_passing_all / (lfp.shape[1] / sampling_rate_lfp)}")
    with open(
        HERE / "results" / "ripples" / f"{mouse}_{imec}.json",
        "w",
    ) as f:
        json.dump(cache_result.model_dump(), f)


def get_lfp_index_sleep_state(
    data_folder,
    n_samples: int,
    sampling_rate_lfp: float,
    plot: bool = False,
):
    """Havent properly tested this yet, but the hacky plot looks fine"""
    seconds, sleep_state = process_sleep_spreadsheet(data_folder)

    assert abs(seconds[-1] - n_samples / sampling_rate_lfp) < 1

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
    assert (len(included_idxs) - n_samples) / sampling_rate_lfp < 1

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

        1 / 0
    return state_idxs


def get_sync(lfp_path: Path, mouse: str, imec: str) -> np.ndarray:
    raw_sync_folder = Path(LOCAL_SSD / "lfp_syncs")
    if (raw_sync_folder / f"npx_sync_times_{mouse}_{imec}.npy").exists():
        return np.load(raw_sync_folder / f"npx_sync_times_{mouse}_{imec}.npy")

    raw_sync_path = raw_sync_folder / f"raw_sync_{mouse}_{imec}.npy"

    if raw_sync_path.exists():
        raw_sync = np.load(raw_sync_path)
    else:
        print("Existing not found, loading raw sync from npyx")
        raw_sync = load_sync_npyx(lfp_path)

    npx_sync_times = threshold_detect(raw_sync, 0.5)
    np.save(raw_sync_folder / f"npx_sync_times_{mouse}_{imec}.npy", npx_sync_times)
    return npx_sync_times


def get_ca1_rsc_channels(lfp_file: Path, df: pd.DataFrame) -> tuple[int, int, int, int]:
    for _, row in df.iterrows():
        if lfp_file.parts[-2] == PureWindowsPath(row["lfp_path"]).parts[-1]:
            return (
                int(row["CA1_Low"]),
                int(row["CA1_High"]),
                int(row["RSC_Low"]),
                int(row["RSC_High"]),
            )
    raise ValueError(f"Could not find channels for {lfp_file.name} in dataframe")


def main() -> None:
    df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)

    umbrella = Path("/Volumes/MarcBusche/Alex/Reactivations/")

    lfp_files = list(umbrella.rglob("*.lf.bin"))
    assert len(lfp_files) > 0, "No LFP files found"

    for lfp_file in lfp_files:

        if "11153" in str(lfp_file):
            print(f"Skipping {lfp_file.name} due to data issues")
            continue

        try:
            ca1_low, ca1_high, rsc_low, rsc_high = get_ca1_rsc_channels(lfp_file, df)

            assert ca1_low < ca1_high < rsc_low < rsc_high
            assert 30 <= ca1_high - ca1_low <= 70
            assert 80 <= rsc_high - rsc_low <= 120

            get_ripple_rate(
                lfp_path=lfp_file.parent,
                region_channels=(ca1_low, ca1_high, rsc_low, rsc_high),
            )
        except Exception as e:
            print(f"Error processing {lfp_file.name}:")
            traceback.print_exc()  # prints the full traceback


def plot_ripple_results():
    results_files = list((HERE / "results" / "ripples").glob("*.json"))
    results: Dict[str, List] = {}
    for result_file in results_files:
        mouse = result_file.name.split("_")[0]
        ripple_cache = RipplesCache.model_validate_json(result_file.read_text())
        passing_checks = (
            np.array(ripple_cache.common_average_reference_check)
            & np.array(ripple_cache.frequency_check)
            & np.array(ripple_cache.super_ripple_check)
        )
        n_seconds = ripple_cache.length_recording
        rate = np.sum(passing_checks) / n_seconds
        if mouse in results:
            results[mouse].append(rate)
        else:
            results[mouse] = [rate]

    to_plot = {"WT": [], "NLGF/S305N": []}
    for mouse, rates in results.items():
        if mouse[:3] == "000":
            print(f"mouse {mouse} is WT")
            to_plot["WT"].append(np.mean(rates))
        else:
            print(f"mouse {mouse} is NLGF/S305N")
            to_plot["NLGF/S305N"].append(np.mean(rates))

    sns.boxplot(to_plot)
    plt.show()


if __name__ == "__main__":
    main()
