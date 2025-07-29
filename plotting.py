from typing import List
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="white", palette="muted")
COLORS = sns.color_palette()


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
    mean = np.mean(arr, 0)
    sem = np.std(arr, 0) / np.sqrt(arr.shape[1])
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
