from pathlib import Path
from typing import List

import numpy as np


import matplotlib.pyplot as plt

from ripples.utils_npyx import load_lfp_reactivations
from ripples.utils import (
    bandpass_filter,
    compute_power,
    compute_envelope,
)

from rsync import Rsync_aligner


from pathlib import Path
import numpy as np
import seaborn as sns


def plot_ripple_power_by_channel(
    lfp_path: Path, mouse: str, aligner: Rsync_aligner
) -> None:
    lfp_cache_path = Path(f"lfp_cache_{mouse}.npy")
    if not lfp_cache_path.exists():
        lfp = load_lfp_reactivations(
            lfp_path,
            chunk_start_seconds=aligner.first_matched_time_A / 2500,
            chunk_end_seconds=aligner.last_matched_time_A / 2500,
        )
        np.save(lfp_cache_path, lfp)
    else:
        lfp = np.load(lfp_cache_path)

    RIPPLE_BAND = [120, 250]
    # Common average referencing
    lfp = lfp - np.mean(lfp, axis=1, keepdims=True)

    # Get the channels with the max SWR channel
    # First 5 min of the recording
    swr_power = compute_power(
        bandpass_filter(
            lfp[:, 0 : 2500 * 5 * 60], RIPPLE_BAND[0], RIPPLE_BAND[1], 2500, order=4
        )
    )
    plt.plot(swr_power)
    plt.xlabel("Channel Number")
    plt.ylabel("SWR Power")
    plt.show()

    # STEVE: There is some useful stuff in this commented out code (also a lot of nonsense) including the theta delta filtering
    # top_channels = np.argsort(swr_power)[-4:]

    # ten_mins = 2500 * 10 * 60

    # if Path(f"lfp_swr_{mouse}.npy").exists():
    #     theta = np.load(f"lfp_theta_{mouse}.npy")
    #     delta = np.load(f"lfp_delta_{mouse}.npy")
    #     swr = np.load(f"lfp_swr_{mouse}.npy")
    # else:
    #     theta = bandpass_filter(lfp[top_channels, :], 6, 9, 2500)
    #     delta = bandpass_filter(lfp[top_channels, :], 0.5, 4, 2500)
    #     swr = bandpass_filter(
    #         lfp[top_channels, :], RIPPLE_BAND[0], RIPPLE_BAND[1], 2500
    #     )

    #     np.save(
    #         Path(f"lfp_theta_{mouse}.npy"),
    #         theta,
    #     )
    #     np.save(
    #         Path(f"lfp_delta_{mouse}.npy"),
    #         delta,
    #     )

    #     np.save(
    #         Path(f"lfp_swr_{mouse}.npy"),
    #         swr,
    #     )
    # # plot delta and theta on two plots

    # def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    #     return np.convolve(arr, np.ones(window), "same") / window

    # def moving_average2(arr: np.ndarray, window: int) -> np.ndarray:
    #     """moving average that downsamples the signal"""
    #     return np.convolve(arr, np.ones(window), "valid") / window

    # theta_envelope = compute_envelope(theta)
    # delta_envelope = compute_envelope(delta)
    # swr_envelope = compute_envelope(swr)

    # plt.plot(moving_average2(theta_envelope[0, :], 25000), color="red", label="Theta")
    # plt.plot(moving_average2(delta_envelope[0, :], 25000), color="blue", label="Delta")
    # plt.plot(moving_average2(swr_envelope[0, :], 25000) * 5, color="green", label="SWR")

    # plt.legend()
    # plt.show()


def plot_spectrogram(lfp: np.ndarray, sampling_rate_lfp: int) -> None:
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

    result = np.vstack(result)
    plt.figure()
    sns.heatmap(
        result,
        square=False,
        # cmap=sns.color_palette("YlOrBr", as_cmap=True),
        cbar_kws={"label": "Log 10 power"},
    )
    plt.ylabel("frequency")
    plt.xlabel("channel")
    plt.xticks(np.arange(384)[::10], np.arange(384)[::10])
    plt.yticks(range(len(edges)), edges)
    plt.show()
