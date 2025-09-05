from dataclasses import dataclass

from pathlib import Path, PureWindowsPath
from typing import Literal, Tuple
import numpy as np
import pandas as pd
from ripples.utils import (
    threshold_detect,
)
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import traceback


from npyx import extract_rawChunk, read_metadata

from ripples.utils_npyx import load_sync_npyx
from consts import KILOSORT_UMBRELLA, LFP_SYNC_FOLDER, LOCAL_SSD
from detect_ripples import detect_ripples
from detect_slow_oscillations import detect_slow_oscillations
from detect_spindles import detect_spindles
from gsheets_importer import gsheet2df
from models import RipplesCache, SlowOscillationCache, SpindleCache

from data_import import Session

import matplotlib.pyplot as plt

from utils import get_data_paths, get_aligners
from collections import namedtuple
from utils import save_figure


HERE = Path(__file__).parent
FIGURE_PATH = HERE / "plots" / "lfp_signatures"

WT_COLOR = "#1f77b4"
NLGF_COLOR = "#ff7f0e"

SHUFFLED_COLOR = sns.color_palette("tab10")[2]


def get_lfp_signatures(
    lfp_path: Path,
    region_channels: Tuple[int, int, int, int],
    session_type: Literal["conditioning", "resting", "tones"],
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

    aligner = (
        aligners[0]
        if session_type == "conditioning"
        else aligners[1] if session_type == "resting" else aligners[2]
    )

    # This should be very close to the sampling rate reported by neuropixels but this is slightly more
    # accurate when aligning to pycontrol. In practice either are probably fine.
    assert abs(aligner.units_B - meta["lowpass"]["sampling_rate"]) < 1
    sampling_rate_lfp = meta["lowpass"]["sampling_rate"]

    lfp_chunk_path = LOCAL_SSD / "lfp_chunks" / f"{mouse}_{imec}_{session_type}.npy"
    if (lfp_chunk_path).exists():
        print("loading existing LFP chunk")
        lfp = np.load(lfp_chunk_path)
    else:
        frame_triggers = np.load(
            frame_trigger_times[0]
            if session_type == "conditioning"
            else (
                frame_trigger_times[1]
                if session_type == "resting"
                else frame_trigger_times[2]
            )
        )
        start_lfp_chunk_seconds, end_lfp_chunk_seconds = (
            aligner.B_to_A(np.array([frame_triggers[0], frame_triggers[-1]]))
            / sampling_rate_lfp
        )
        assert (
            start_lfp_chunk_seconds < end_lfp_chunk_seconds
        ), "Start time must be before end time"
        assert 10 * 60 < end_lfp_chunk_seconds - start_lfp_chunk_seconds < 40 * 60
        lfp = extract_rawChunk(
            lfp_path,
            [
                start_lfp_chunk_seconds,
                end_lfp_chunk_seconds,
            ],  # now taking the recording length as a float
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

        np.save(lfp_chunk_path, lfp)

    # plot_lfp_profile(
    #     lfp,
    #     ripple_band=RIPPLE_BAND,
    #     sampling_rate_lfp=sampling_rate_lfp,
    #     mouse=mouse,
    #     imec=imec,
    # )

    detect_ripples(
        mouse,
        imec,
        ca1_low,
        ca1_high,
        data_folder,
        sampling_rate_lfp,
        lfp,
        session_type=session_type,
    )

    # lfp_spindle, max_power_channel = detect_spindles(
    #     mouse, imec, rsc_low, rsc_high, data_folder, sampling_rate_lfp, lfp
    # )
    # detect_slow_oscillations(
    #     lfp_spindle, max_power_channel, sampling_rate_lfp, mouse, imec, data_folder
    # )


def get_sync(lfp_path: Path, mouse: str, imec: str) -> np.ndarray:
    raw_sync_path = LFP_SYNC_FOLDER / f"raw_sync_{mouse}_{imec}.npy"
    processed_sync_path = LFP_SYNC_FOLDER / f"npx_sync_times_{mouse}_{imec}.npy"
    if processed_sync_path.exists():
        return np.load(processed_sync_path)

    if raw_sync_path.exists():
        raw_sync = np.load(raw_sync_path)
    else:
        print("Existing not found, loading raw sync from npyx")
        raw_sync = load_sync_npyx(lfp_path, "lowpass")

    npx_sync_times = threshold_detect(raw_sync, 0.5)
    # There are a few random sync pulses at the start from failed pycontrol sessions
    if mouse == "11150":
        npx_sync_times = npx_sync_times[-5435:]

    np.save(processed_sync_path, npx_sync_times)
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
    lfp_files = list(KILOSORT_UMBRELLA.rglob("*.lf.bin"))
    assert len(lfp_files) > 0, "No LFP files found"

    for lfp_file in lfp_files:
        if "11153" in str(lfp_file):
            print(f"Skipping {lfp_file.name} due to data issues")
            continue

        ca1_low, ca1_high, rsc_low, rsc_high = get_ca1_rsc_channels(lfp_file, df)

        assert ca1_low < ca1_high < rsc_low < rsc_high
        assert 30 <= ca1_high - ca1_low <= 70
        assert 80 <= rsc_high - rsc_low <= 120

        get_lfp_signatures(
            lfp_path=lfp_file.parent,
            region_channels=(ca1_low, ca1_high, rsc_low, rsc_high),
            session_type="conditioning",
        )


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

    summary_df = df.groupby(["Genotype", "Sleep State"]).agg(
        {
            "Ripple rate (Hz)": ["mean", "median", "std"],
            "Ripple duration (ms)": ["mean", "median", "std"],
            "Ripple amplitude (µV)": ["mean", "median", "std"],
        }
    )
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()

    # Store p values in a dataframe
    p_values_data = {"Measure": [], "Sleep_State": [], "P_Value": [], "U_Statistic": []}
    df = df.replace({"Sleep State": {"awake": "Awake", "nrem": "NREM"}})

    for key in ["Ripple duration (ms)", "Ripple rate (Hz)", "Ripple amplitude (µV)"]:
        plt.figure()
        awake_p, n_rem_p = get_p_values(df, key)

        # Store p-values and U statistics
        p_values_data["Measure"].extend([key, key])
        p_values_data["Sleep_State"].extend(["awake", "nrem"])
        p_values_data["P_Value"].extend([awake_p.pvalue, n_rem_p.pvalue])
        p_values_data["U_Statistic"].extend([awake_p.statistic, n_rem_p.statistic])

        b = sns.boxplot(
            data=df,
            x="Sleep State",
            y=key,
            hue="Genotype",
            palette={"WT": WT_COLOR, "NLGF/S305N": NLGF_COLOR},
            showfliers=False,
            legend=False,
        )
        b.tick_params(labelsize=12)
        b.set_xlabel("Sleep State", fontsize=14, fontweight="bold")
        b.set_ylabel(key, fontsize=14, fontweight="bold")
        plt.grid(axis="y")

        sns.stripplot(
            data=df,
            x="Sleep State",
            y=key,
            hue="Genotype",
            palette={"WT": WT_COLOR, "NLGF/S305N": NLGF_COLOR},
            dodge=True,
            alpha=1,
            legend=False,
            linewidth=0.5,
        )

        plt.title(
            (
                "Ripple Duration Per State"
                if "duration" in key
                else (
                    "Ripple Rate Per State"
                    if "rate" in key
                    else "Ripple Amplitude Per State"
                )
            ),
            fontsize=16,
            fontweight="bold",
        )
        if key == "Ripple duration (ms)":
            plt.ylim(0, 80)
        if key == "Ripple amplitude (µV)":
            plt.ylim(0, 40)
        save_figure(key, FIGURE_PATH)

    # Create p-values dataframe
    p_values_df = pd.DataFrame(p_values_data)

    summary_df.to_csv(HERE / "results" / "ripples" / "ripple_summary.csv")
    p_values_df.to_csv(
        HERE / "results" / "ripples" / "ripple_p_values.csv", index=False
    )


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

    # Store p values in a dataframe
    p_values_data = {"Measure": [], "Sleep_State": [], "P_Value": [], "U_Statistic": []}
    df = df.replace({"Sleep State": {"awake": "Awake", "nrem": "NREM"}})

    for key in [
        "Spindle duration (ms)",
        "Spindle rate (min$^{-1}$)",
        "Spindle amplitude (µV)",
    ]:
        awake_p, n_rem_p = get_p_values(df, key)

        # Store p-values and U statistics
        p_values_data["Measure"].extend([key, key])
        p_values_data["Sleep_State"].extend(["awake", "nrem"])
        p_values_data["P_Value"].extend([awake_p.pvalue, n_rem_p.pvalue])
        p_values_data["U_Statistic"].extend([awake_p.statistic, n_rem_p.statistic])

        plt.figure()
        b = sns.boxplot(
            data=df,
            x="Sleep State",
            y=key,
            hue="Genotype",
            palette={"WT": WT_COLOR, "NLGF/S305N": NLGF_COLOR},
            showfliers=False,
            legend=False,
        )
        b.tick_params(labelsize=12)
        b.set_xlabel("Sleep State", fontsize=14, fontweight="bold")
        b.set_ylabel(key, fontsize=14, fontweight="bold")
        plt.grid(axis="y")

        sns.stripplot(
            data=df,
            x="Sleep State",
            y=key,
            hue="Genotype",
            palette={"WT": WT_COLOR, "NLGF/S305N": NLGF_COLOR},
            dodge=True,
            alpha=1,
            legend=False,
            linewidth=0.5,
        )

        plt.title(
            (
                "Spindle Duration Per State"
                if "duration" in key
                else (
                    "Spindle Rate Per State"
                    if "rate" in key
                    else "Spindle Amplitude Per State"
                )
            ),
            fontsize=16,
            fontweight="bold",
        )
        if key == "Spindle duration (ms)":
            plt.ylim(400, 1000)
        if key == "Spindle amplitude (µV)":
            plt.ylim(0, 20)
        save_figure(key, FIGURE_PATH)

    # Create p-values dataframe
    p_values_df = pd.DataFrame(p_values_data)

    # Save summary and p-values
    summary_df = df.groupby(["Genotype", "Sleep State"]).agg(
        {
            "Spindle rate (min$^{-1}$)": ["mean", "median", "std"],
            "Spindle duration (ms)": ["mean", "median", "std"],
            "Spindle amplitude (µV)": ["mean", "median", "std"],
        }
    )
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()

    summary_df.to_csv(HERE / "results" / "spindles" / "spindle_summary.csv")
    p_values_df.to_csv(
        HERE / "results" / "spindles" / "spindle_p_values.csv", index=False
    )


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

    # Store p values in a dataframe
    p_values_data = {"Measure": [], "Sleep_State": [], "P_Value": [], "U_Statistic": []}
    df = df.replace({"Sleep State": {"awake": "Awake", "nrem": "NREM"}})

    for key in [
        "Slow Oscillation rate (min$^{-1}$)",
        "Slow Oscillation duration (ms)",
        "Slow Oscillation amplitude (µV)",
    ]:
        plt.figure()
        b = sns.boxplot(
            data=df,
            x="Sleep State",
            y=key,
            hue="Genotype",
            palette={"WT": WT_COLOR, "NLGF/S305N": NLGF_COLOR},
            showfliers=False,
            legend=False,
        )
        b.tick_params(labelsize=12)
        b.set_xlabel("Sleep State", fontsize=14, fontweight="bold")
        b.set_ylabel(key, fontsize=14, fontweight="bold")
        plt.grid(axis="y")

        sns.stripplot(
            data=df,
            x="Sleep State",
            y=key,
            hue="Genotype",
            palette={"WT": WT_COLOR, "NLGF/S305N": NLGF_COLOR},
            dodge=True,
            alpha=1,
            legend=False,
            linewidth=0.5,
        )

        awake_p, n_rem_p = get_p_values(df, key)

        # Store p-values and U statistics
        p_values_data["Measure"].extend([key, key])
        p_values_data["Sleep_State"].extend(["awake", "nrem"])
        p_values_data["P_Value"].extend([awake_p.pvalue, n_rem_p.pvalue])
        p_values_data["U_Statistic"].extend([awake_p.statistic, n_rem_p.statistic])

        plt.title(
            (
                "Slow Oscillation Duration Per State"
                if "duration" in key
                else (
                    "Slow Oscillation Rate Per State"
                    if "rate" in key
                    else "Slow Oscillation Amplitude Per State"
                )
            ),
            fontsize=16,
            fontweight="bold",
        )
        save_figure(key, FIGURE_PATH)

    # Create p-values dataframe
    p_values_df = pd.DataFrame(p_values_data)

    summary_df = df.groupby(["Genotype", "Sleep State"]).agg(
        {
            "Slow Oscillation rate (min$^{-1}$)": ["mean", "median", "std"],
            "Slow Oscillation duration (ms)": ["mean", "median", "std"],
            "Slow Oscillation amplitude (µV)": ["mean", "median", "std"],
        }
    )
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()

    summary_df.to_csv(
        HERE / "results" / "slow_oscillations" / "slow_oscillation_summary.csv"
    )
    p_values_df.to_csv(
        HERE / "results" / "slow_oscillations" / "slow_oscillation_p_values.csv",
        index=False,
    )


MannwhitneyuResult = namedtuple("MannwhitneyuResult", ("statistic", "pvalue"))


def get_p_values(df, results_key) -> tuple[MannwhitneyuResult, MannwhitneyuResult]:
    wt = df[(df["Genotype"] == "WT")]
    nlgf = df[(df["Genotype"] == "NLGF/S305N")]

    # Handle both original and replaced sleep state names
    awake_states = ["awake", "Awake"]
    nrem_states = ["nrem", "NREM"]

    wt_awake = wt[wt["Sleep State"].isin(awake_states)][results_key].to_numpy()
    nlgf_awake = nlgf[nlgf["Sleep State"].isin(awake_states)][results_key].to_numpy()

    wt_nrem = wt[wt["Sleep State"].isin(nrem_states)][results_key].to_numpy()
    nlgf_nrem = nlgf[nlgf["Sleep State"].isin(nrem_states)][results_key].to_numpy()

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


def get_baseline_events(
    spindle_times: np.ndarray, ripple_times: np.ndarray, distance_seconds: float
):
    diffs = np.abs(spindle_times[:, None] - ripple_times[None, :])

    # Mask spindles that are farther than distance_seconds from all ripples
    mask = np.all(diffs > distance_seconds, axis=1)
    return spindle_times[mask]


def get_peak_times_for_coupling(
    cache: RipplesCache | SpindleCache | SlowOscillationCache, mouse: str
) -> np.ndarray:
    match cache.__class__.__name__:
        case "RipplesCache":
            passing_checks = (
                np.array(cache.common_average_reference_check)
                & np.array(cache.frequency_check)
                & np.array(cache.super_ripple_check)
            )
            if mouse == "00053":
                passing_checks = passing_checks[: len(cache.candidate_events)]

            ripples = np.array(cache.candidate_events)[passing_checks]
            ripple_states = np.array(cache.state)[passing_checks]

            nrem_ripples = ripples[ripple_states == "nrem"]
            return np.array([ripple.peak_idx for ripple in nrem_ripples])

        case "SpindleCache":
            nrem_spindles = np.array(cache.spindles)[np.array(cache.state) == "nrem"]
            return np.array([spindle.peak_idx for spindle in nrem_spindles])

        case "SlowOscillationCache":
            downsampled_lfp = np.array(cache.downsampled_lfp)
            state = np.array(cache.state)
            starts = np.array(cache.starts)[state == "nrem"]
            ends = np.array(cache.ends)[state == "nrem"]
            troughs = []

            for start, end in zip(starts, ends, strict=True):
                lfp = downsampled_lfp[
                    start // cache.downsample_factor : end // cache.downsample_factor
                ]
                trough_downsampled = np.argmin(lfp) + start // cache.downsample_factor
                troughs.append(trough_downsampled * cache.downsample_factor)
            return np.array(troughs)

        case _:
            raise ValueError(f"Unsupported cache type: {cache.__class__.__name__}")


def get_coupling_matrix(
    cache1: RipplesCache | SpindleCache | SlowOscillationCache,
    cache2: RipplesCache | SpindleCache | SlowOscillationCache,
    mouse: str,
    bins: np.ndarray,
    remove_events_from_baseline: bool,
) -> np.ndarray:
    times1 = get_peak_times_for_coupling(cache1, mouse)
    times2 = get_peak_times_for_coupling(cache2, mouse)

    if 0 in {len(times1), len(times2)}:
        print("No events found")
        return None

    coupling_matrix = np.array(
        [
            np.histogram(
                (times2 - time1) / 2500,
                bins=bins,
                range=(bins[0], bins[-1]),
            )[0]
            for time1 in times1
        ]
    )

    if remove_events_from_baseline:
        removal_distance = 0.1 * 2500

        baseline_events = get_baseline_events(
            times2, times1, distance_seconds=removal_distance
        )
        baseline_rate = len(baseline_events) / (
            (cache1.state_lengths["nrem"] - (len(times1) * removal_distance * 2)) / 2500
        )
        print(baseline_rate * 60)
    else:
        baseline_rate = len(times2) / (cache2.state_lengths["nrem"] / 2500)

    coupling_matrix = (coupling_matrix * (1 / (bins[1] - bins[0]))) / baseline_rate

    return coupling_matrix


def get_coupling_results():
    ripple_files = list((HERE / "results" / "ripples").glob("*.json"))

    wt_result = []
    nlgf_result = []

    bin_size = 25 / 1000
    bins = np.arange(-2, 2 + bin_size, bin_size)

    for ripple_file in ripple_files:
        mouse = ripple_file.name.split("_")[0]
        imec = ripple_file.stem[-1]
        spindle_file = HERE / "results" / "spindles" / f"{mouse}_imec_{imec}.json"
        slow_oscillation_file = (
            HERE / "results" / "slow_oscillations" / f"{mouse}_imec_{imec}.json"
        )
        assert spindle_file.exists(), f"Spindle file {spindle_file} does not exist"
        assert spindle_file.stem == ripple_file.stem
        coupling_matrix = get_coupling_matrix(
            SlowOscillationCache.model_validate_json(slow_oscillation_file.read_text()),
            # SpindleCache.model_validate_json(spindle_file.read_text()),
            RipplesCache.model_validate_json(ripple_file.read_text()),
            mouse,
            bins=bins,
            remove_events_from_baseline=False,
        )

        if coupling_matrix is None:
            continue

        if mouse[:3] == "000":
            print(f"Mouse  {mouse} is WT")
            wt_result.append(coupling_matrix)
            # wt_n_spindles += n_baseline_spindles
            # wt_length_nrem += length_nrem
        else:
            print(f"Mouse  {mouse} is NLGF")
            nlgf_result.append(coupling_matrix)
            # nlgf_n_spindles += n_baseline_spindles
            # nlgf_length_nrem += length_nrem

    wt = np.concatenate(wt_result)
    nlgf = np.concatenate(nlgf_result)

    wt_sum = np.mean(wt, axis=0)
    nlgf_sum = np.mean(nlgf, axis=0)

    bin_centers = bins[:-1] + bin_size / 2

    sigma_seconds = 0.1
    sigma_bins = sigma_seconds / bin_size
    plt.plot(
        bin_centers,
        gaussian_filter1d(wt_sum, sigma=sigma_bins),
        color=WT_COLOR,
        label="WT",
    )
    plt.plot(
        bin_centers,
        gaussian_filter1d(nlgf_sum, sigma=sigma_bins),
        color=NLGF_COLOR,
        label="NLGFxS305N",
    )

    plt.legend()
    plt.xlim(-1.5, 1.5)
    plt.xlabel("Time from Slow Oscillation Trough (s)", fontsize=14, fontweight="bold")
    plt.ylabel(
        "Ripple Rate (fold change from baseline)",
        fontsize=14,
        fontweight="bold",
    )

    # shaded_line_plot(arr=wt, x_axis=bin_centers, color=WT_COLOR, label="WT")
    # shaded_line_plot(arr=nlgf, x_axis=bin_centers, color=NLGF_COLOR, label="NLGFxS305N")
    # plt.show()


if __name__ == "__main__":
    main()
    # get_coupling_results()
    # plot_spindle_results()
