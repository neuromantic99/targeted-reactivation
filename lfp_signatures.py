from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from ripples.utils import (
    threshold_detect,
)
from scipy import stats
import seaborn as sns
import traceback


from npyx import extract_rawChunk, read_metadata

from ripples.utils_npyx import load_sync_npyx
from consts import LOCAL_SSD
from detect_ripples import detect_ripples
from detect_slow_oscillations import detect_slow_oscillations
from detect_spindles import detect_spindles
from gsheets_importer import gsheet2df
from models import RipplesCache, SlowOscillationCache, SpindleCache

from data_import import Session
from main import HERE, get_aligners

import matplotlib.pyplot as plt

from utils import get_data_paths

HERE = Path(__file__).parent

FIGURE_PATH = HERE / "plots" / "lfp_signatures"


def get_lfp_signatures(
    lfp_path: Path, region_channels: Tuple[int, int, int, int]
) -> None:
    mouse = lfp_path.parent.parent.name
    imec = f"imec_{str(lfp_path).split('imec')[1]}"

    if (HERE / "results" / "slow_oscillations" / f"{mouse}_{imec}.json").exists():
        print(f"Slow oscillation results for {mouse}_{imec} already exist, skipping.")
        return

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

    resting_lfp_path = LOCAL_SSD / f"resting_lfps/{mouse}_{imec}.npy"
    if (resting_lfp_path).exists():
        print("loading existing resting LFP")
        lfp = np.load(resting_lfp_path)
    else:
        frame_triggers = np.load(frame_trigger_times[1])
        start_rest, end_rest = (
            aligners[1].B_to_A(np.array([frame_triggers[0], frame_triggers[-1]]))
            / sampling_rate_lfp
        )
        assert start_rest < end_rest, "Start time must be before end time"
        assert 10 * 60 > end_rest - start_rest > 40 * 60
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
    lfp_spindle, max_power_channel = detect_spindles(
        mouse, imec, rsc_low, rsc_high, data_folder, sampling_rate_lfp, lfp
    )
    detect_slow_oscillations(
        lfp_spindle, max_power_channel, sampling_rate_lfp, mouse, imec, data_folder
    )


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


# lfp_files = [
#     Path(
#         "/Volumes/MarcBusche/Alex/Reactivations/2025-05-20/11150/20250520_g0/20250520_g0_imec0/20250520_g0_t0.imec0.lf.bin"
#     )
# ]


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

            get_lfp_signatures(
                lfp_path=lfp_file.parent,
                region_channels=(ca1_low, ca1_high, rsc_low, rsc_high),
            )
        except Exception as e:
            print(f"Error processing {lfp_file.name}:")
            traceback.print_exc()  # prints the full traceback


def plot_ripple_results():
    results_files = list((HERE / "results" / "ripples").glob("*.json"))
    data = {
        "Genotype": [],
        "mouse_id": [],
        "Sleep State": [],
        "Ripple rate (Hz)": [],
        "Ripple duration (ms)": [],
        "Ripple amplitude (µV)": [],
    }

    for result_file in results_files:
        mouse = result_file.name.split("_")[0]

        ripple_cache = RipplesCache.model_validate_json(result_file.read_text())
        passing_checks = (
            np.array(ripple_cache.common_average_reference_check)
            & np.array(ripple_cache.frequency_check)
            & np.array(ripple_cache.super_ripple_check)
        )
        if mouse == "00053":
            passing_checks = passing_checks[: len(ripple_cache.candidate_events)]

        ripples = np.array(ripple_cache.candidate_events)[passing_checks]
        ripple_states = np.array(ripple_cache.state)[passing_checks]

        for state in np.unique(ripple_states):
            if state in {"transition", "rem"}:
                continue
            state_ripples = ripples[ripple_states == state]
            state_length = ripple_cache.state_lengths[state] / 2500
            data["Genotype"].append("WT" if mouse[:3] == "000" else "NLGF/S305N")
            data["mouse_id"].append(mouse)
            data["Sleep State"].append(state)
            data["Ripple rate (Hz)"].append(len(state_ripples) / state_length)
            data["Ripple duration (ms)"].append(
                np.mean(
                    [(ripple.offset - ripple.onset) / 2500 for ripple in state_ripples]
                )
                * 1000
            )
            data["Ripple amplitude (µV)"].append(
                np.mean([ripple.peak_amplitude for ripple in state_ripples])
            )

    df = pd.DataFrame(data)
    # Mean the ripple rate within a mouse and state
    df = df.groupby(["Genotype", "mouse_id", "Sleep State"]).mean().reset_index()

    for key in ["Ripple duration (ms)", "Ripple rate (Hz)", "Ripple amplitude (µV)"]:
        plt.figure()
        awake_p, n_rem_p = get_p_values(df, key)
        sns.boxplot(data=df, x="Sleep State", y=key, hue="Genotype")
        plt.title(
            f"Mann-Whitney U Test Results: awake: {round(awake_p.pvalue, 2)}, nrem: {round(n_rem_p.pvalue, 2)}"
        )
        if key == "Ripple duration (ms)":
            plt.ylim(0, 80)
        if key == "Ripple amplitude (µV)":
            plt.ylim(0, 40)
        save_figure(key, FIGURE_PATH)


def plot_spindle_results():
    results_files = list((HERE / "results" / "spindles").glob("*.json"))
    data = {
        "Genotype": [],
        "mouse_id": [],
        "Spindle rate (min$^{-1}$)": [],
        "Spindle duration (ms)": [],
        "Spindle amplitude (µV)": [],
        "Sleep State": [],
    }
    for result_file in results_files:
        mouse = result_file.name.split("_")[0]
        spindle_cache = SpindleCache.model_validate_json(result_file.read_text())
        spindles = np.array(spindle_cache.spindles)
        spindle_states = np.array(spindle_cache.state)

        for state in np.unique(spindle_states):
            if state in {"transition", "rem"}:
                continue
            state_spindles = spindles[spindle_states == state]
            state_length = spindle_cache.state_lengths[state] / 2500
            data["Genotype"].append("WT" if mouse[:3] == "000" else "NLGF/S305N")
            data["mouse_id"].append(mouse)
            data["Sleep State"].append(state)
            data["Spindle rate (min$^{-1}$)"].append(
                (len(state_spindles) / state_length) * 60
            )
            data["Spindle duration (ms)"].append(
                np.mean(
                    [
                        (spindle.offset - spindle.onset) / 2500
                        for spindle in state_spindles
                    ]
                )
                * 1000
            )
            data["Spindle amplitude (µV)"].append(
                np.mean([spindle.peak_amplitude for spindle in state_spindles])
            )

    df = pd.DataFrame(data)
    # Average across probes within a mouse
    df = df.groupby(["Genotype", "mouse_id", "Sleep State"]).mean().reset_index()

    for key in [
        "Spindle duration (ms)",
        "Spindle rate (min$^{-1}$)",
        "Spindle amplitude (µV)",
    ]:
        awake_p, n_rem_p = get_p_values(df, key)

        plt.figure()
        sns.boxplot(data=df, x="Sleep State", y=key, hue="Genotype")
        plt.title(
            f"Mann-Whitney U Test Results: awake: {round(awake_p.pvalue, 2)}, nrem: {round(n_rem_p.pvalue, 2)}"
        )
        if key == "Spindle duration (ms)":
            plt.ylim(0, 1000)
        if key == "Spindle amplitude (µV)":
            plt.ylim(0, 20)
        save_figure(key, FIGURE_PATH)


def plot_slow_oscillation_results():
    results_files = list((HERE / "results" / "slow_oscillations").glob("*.json"))
    data = {
        "Genotype": [],
        "mouse_id": [],
        "Slow Oscillation rate (min$^{-1}$)": [],
        "Slow Oscillation duration (ms)": [],
        "Slow Oscillation amplitude (µV)": [],
        "Sleep State": [],
    }
    for result_file in results_files:
        if "medianRef" in str(result_file):
            print(f"Skipping {result_file} due to medianRef")
            continue
        mouse = result_file.name.split("_")[0]
        slow_cache = SlowOscillationCache.model_validate_json(result_file.read_text())
        slow_starts = np.array(slow_cache.starts)
        slow_ends = np.array(slow_cache.ends)
        slow_states = np.array(slow_cache.state)

        for state in np.unique(slow_states):
            if state in {"transition", "rem"}:
                continue
            state_start = slow_starts[slow_states == state]
            state_end = slow_ends[slow_states == state]
            state_length = slow_cache.state_lengths[state] / 2500
            data["Genotype"].append("WT" if mouse[:3] == "000" else "NLGF/S305N")
            data["mouse_id"].append(mouse)
            data["Sleep State"].append(state)
            data["Slow Oscillation rate (min$^{-1}$)"].append(
                (len(state_start) / state_length) * 60
            )
            data["Slow Oscillation duration (ms)"].append(
                np.mean(
                    [(end - start) / 2500 for start, end in zip(state_start, state_end)]
                )
                * 1000
            )
            data["Slow Oscillation amplitude (µV)"].append(
                np.mean(
                    [
                        max(
                            slow_cache.downsampled_lfp[
                                start
                                // slow_cache.downsample_factor : end
                                // slow_cache.downsample_factor
                            ]
                        )
                        - min(
                            slow_cache.downsampled_lfp[
                                start
                                // slow_cache.downsample_factor : end
                                // slow_cache.downsample_factor
                            ]
                        )
                        for start, end in zip(state_start, state_end)
                    ]
                )
            )

    df = pd.DataFrame(data)
    # Average across probes within a mouse
    df = df.groupby(["Genotype", "mouse_id", "Sleep State"]).mean().reset_index()

    for key in [
        "Slow Oscillation rate (min$^{-1}$)",
        "Slow Oscillation duration (ms)",
        "Slow Oscillation amplitude (µV)",
    ]:
        plt.figure()
        sns.boxplot(data=df, x="Sleep State", y=key, hue="Genotype")
        awake_p, n_rem_p = get_p_values(df, key)
        plt.title(
            f"Mann-Whitney U Test Results: awake: {round(awake_p.pvalue, 2)}, nrem: {round(n_rem_p.pvalue, 2)}"
        )
        save_figure(key, FIGURE_PATH)


def get_p_values(df, results_key) -> tuple[float, float]:

    wt = df[(df["Genotype"] == "WT")]
    nlgf = df[(df["Genotype"] == "NLGF/S305N")]

    wt_awake = wt[wt["Sleep State"] == "awake"][results_key].to_numpy()
    nlgf_awake = nlgf[nlgf["Sleep State"] == "awake"][results_key].to_numpy()

    wt_nrem = wt[wt["Sleep State"] == "nrem"][results_key].to_numpy()
    nlgf_nrem = nlgf[nlgf["Sleep State"] == "nrem"][results_key].to_numpy()

    awake_p = stats.mannwhitneyu(
        wt_awake,
        nlgf_awake,
        alternative="two-sided",
    )

    n_rem_p = stats.mannwhitneyu(
        wt_nrem,
        nlgf_nrem,
        alternative="two-sided",
    )

    return awake_p, n_rem_p


def permutation_test(x, y, alternative=None):
    n_simulations = 1000
    n_x = len(x)
    simulated_diffs = []

    all_data = np.concatenate([x, y])

    for i in range(n_simulations):
        users_shuffled = np.random.permutation(all_data)
        simulated_x = users_shuffled[:n_x]
        simulated_y = users_shuffled[n_x:]
        simulated_diff = np.mean(simulated_x) - np.mean(simulated_y)
        simulated_diffs.append(simulated_diff)

    real = np.mean(x) - np.mean(y)
    p_value = np.mean(np.abs(simulated_diffs) >= np.abs(real))

    @dataclass
    class Result:
        pvalue: float

    return Result(pvalue=p_value)


def save_figure(name: str, path: Path):
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(path / f"{name}.pdf", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    plot_ripple_results()
    plot_spindle_results()
    plot_slow_oscillation_results()
    plt.show()
    # main()
