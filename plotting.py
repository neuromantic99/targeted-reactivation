from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


from ripples.utils import compute_power, bandpass_filter

from consts import LOCAL_SSD


sns.set_theme(style="white", palette="muted")
COLORS = sns.color_palette()

here = Path(__file__).parent


def plot_individual_cell_results(
    result: np.ndarray,
    stim_labels: np.ndarray,
    sampling_rate: int,
    n_bins: int,
    bin_edges: np.ndarray,
) -> None:
    # Summed across trials

    mask = stim_labels == 3000
    result = result[mask, :, :]
    result = np.sum(result, axis=0)

    baseline = np.mean(result[:, : n_bins // 2], axis=1)
    result -= baseline[:, np.newaxis]

    # bin_width = (bin_edges[1] - bin_edges[0]) / sampling_rate
    # result /= bin_width

    center = n_bins // 2

    pre = np.mean(result[:, center - int(0.1 * sampling_rate) : center], axis=1)
    post = np.mean(result[:, center : center + int(0.2 * sampling_rate)], axis=1)
    change = post - pre

    sort_idx = np.argsort(change)[::-1]
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure()
    for cell_idx in sort_idx[:5]:
        plt.plot(
            midpoints / sampling_rate, result[cell_idx, :], alpha=0.5, color="black"
        )

    plt.xlabel("Time from stimulus onset (s)")
    plt.ylabel("Spikes per cell (summed across trials)")

    plt.figure()
    for cell_idx in sort_idx[-5:]:
        plt.plot(
            midpoints / sampling_rate, result[cell_idx, :], alpha=0.5, color="black"
        )

    plt.xlabel("Time from stimulus onset (s)")
    plt.ylabel("Spikes per cell (summed across trials)")


def plot_cell_summed_results(
    trial_array: np.ndarray,
    stim_labels: np.ndarray,
    sampling_rate: int,
    n_bins: int,
    bin_edges: np.ndarray,
) -> None:
    trial_array = np.sum(trial_array, axis=1)
    baseline = np.mean(trial_array[:, : n_bins // 2], axis=1)
    trial_array -= baseline[:, np.newaxis]

    bin_width = (bin_edges[1] - bin_edges[0]) / sampling_rate
    trial_array /= bin_width

    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure()

    for idx, stim_type in enumerate(np.unique(stim_labels)):
        mask = stim_labels == stim_type

        shaded_line_plot(
            trial_array[mask, :],
            midpoints / sampling_rate,
            color=COLORS[idx],
            label=stim_type,
        )

    plt.legend()
    plt.xlabel("Time from stimulus onset (s)")
    plt.ylabel("Total spike rate (Hz)")
    plt.axvline(0, color="red", linestyle="--", alpha=0.5)
    sns.despine()


def shaded_line_plot(
    arr: np.ndarray,
    x_axis: np.ndarray | List[float],
    color: str,
    label: str,
) -> None:
    mean = np.nanmean(arr, 0)
    sem = np.nanstd(arr, 0) / np.sqrt(arr.shape[1])
    plt.plot(x_axis, mean, color=color, label=label, marker="", zorder=1)
    plt.fill_between(
        x_axis,
        np.subtract(
            mean,
            sem,
        ),
        np.add(
            mean,
            sem,
        ),
        alpha=0.2,
        color=color,
    )


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(arr, np.ones(window), "same") / window


def plot_lfp_profile(
    lfp: np.ndarray,
    ripple_band: List[int],
    sampling_rate_lfp: int,
    mouse: str,
    imec: str,
) -> None:

    ripple_power_whole_probe = compute_power(
        bandpass_filter(
            lfp[:, : 2500 * 30],
            ripple_band[0],
            ripple_band[1],
            sampling_rate_lfp,
            order=4,
        )
    )

    theta = compute_power(
        bandpass_filter(
            lfp[:, : 2500 * 30],
            5,
            9,
            sampling_rate_lfp,
            order=4,
        )
    )

    mua = np.load(LOCAL_SSD / "MUA_depths" / f"{mouse}_{imec}.npy")
    mua = moving_average(mua, 10)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("Normalised LFP power", color="black")
    ax2 = ax1.twinx()
    ax1.plot(
        ripple_power_whole_probe / max(ripple_power_whole_probe),
        color="blue",
        label="Ripple power",
    )
    ax1.plot(theta / max(theta), color="green", label="Theta power")
    ax1.legend(loc="upper right")
    ax2.plot(mua, color="red", label="MUA")
    ax2.set_ylabel("MUA spikes", color="red")

    title = f"{mouse}_{imec}"
    plt.title(title)
    plt.savefig(here / "plots" / "trajectory" / f"{title}.png")


def plot_lfp_spectrogram(
    lfp: np.ndarray,
    sampling_rate_lfp: float,
) -> None:

    max_freq = 550
    edges = (
        list(range(2, 10, 1))
        + list(range(10, 100, 10))
        + list(range(100, max_freq, 50))
    )

    result = []
    for idx in range(len(edges) - 1):
        start = edges[idx]
        end = edges[idx + 1]

        result.append(
            compute_power(bandpass_filter(lfp, start, end, sampling_rate_lfp, order=4))
        )

    result = np.array(result).T
    result = np.log(result)
    result = np.flipud(result)

    plt.figure()
    sns.heatmap(
        result,
        square=False,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        cbar_kws={"label": "Log power"},
    )
    plt.xticks(range(len(edges)), edges)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Channel")

    plt.show()
