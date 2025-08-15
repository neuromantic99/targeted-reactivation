import json
from pathlib import Path
from typing import Tuple
import numpy as np

from ripples.utils import (
    compute_power,
    bandpass_filter,
    threshold_detect,
    compute_envelope,
    get_event_frequency,
)
from scipy import signal

import matplotlib.pyplot as plt

from models import SlowOscillationCache
from utils import get_lfp_index_sleep_state

HERE = Path(__file__).parent


def detect_slow_oscillations(
    lfp: np.ndarray,
    max_power_channel: int,
    sampling_rate_lfp: float,
    mouse: str,
    imec: str,
    data_folder: Path,
) -> None:
    """
    Detect slow oscillations in the spindle LFP data.
    """

    filtered = bandpass_filter(lfp, 0.3, 1.25, sampling_rate_lfp, order=2)

    # Should this be the absolute?
    # hilbert_transformed = signal.hilbert(filtered, axis=1)
    # angle = np.angle(hilbert_transformed[max_power_channel, :])

    slow_oscillation = filtered[max_power_channel, :]
    crossings = threshold_detect_falling_edge(slow_oscillation, 0)
    valid_event_start = np.where(length_check(crossings, sampling_rate_lfp))[0]

    starts = crossings[valid_event_start]
    ends = crossings[valid_event_start + 1]
    assert max((ends - starts) / sampling_rate_lfp) < 2.0
    assert min((ends - starts) / sampling_rate_lfp) > 0.8
    valid_idx = amplitude_check(slow_oscillation, starts, ends)

    starts = starts[valid_idx]
    ends = ends[valid_idx]

    # Make sure didn't introduce some sort of off-by-one
    assert max((ends - starts) / sampling_rate_lfp) < 2.0
    assert min((ends - starts) / sampling_rate_lfp) > 0.8

    print(
        f"Overall slow oscillation rate {60 * (len(starts) / (lfp.shape[1] / sampling_rate_lfp)):.2f} per minute"
    )

    lfp_state_idx = get_lfp_index_sleep_state(
        data_folder=data_folder,
        # This mouse was done with buffering rather than streaming.
        # Need to remove ripples past the 30 minute mark below
        n_samples=lfp.shape[1] if mouse != "00053" else 30 * 60 * sampling_rate_lfp,
        sampling_rate_lfp=sampling_rate_lfp,
    )

    if mouse == "00053":
        keep = starts < 30 * 60 * sampling_rate_lfp
        starts = starts[keep]
        ends = ends[keep]

    oscillation_state = []

    for start in starts:
        for state, idxs in lfp_state_idx.items():
            if start in idxs:
                oscillation_state.append(state)
                break

    assert len(oscillation_state) == len(starts)

    cache_result = SlowOscillationCache(
        starts=starts.tolist(),
        ends=ends.tolist(),
        state=oscillation_state,
        state_lengths={state: len(list(idxs)) for state, idxs in lfp_state_idx.items()},
        downsampled_lfp=slow_oscillation[::10].tolist(),
        downsample_factor=10,
    )

    with open(
        HERE / "results" / "slow_oscillations" / f"{mouse}_{imec}.json",
        "w",
    ) as f:
        json.dump(cache_result.model_dump(), f)


def amplitude_check(
    slow_oscillation: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    trough_to_peak = np.array([])
    trough_amplitude = np.array([])
    for start, end in zip(starts, ends, strict=True):
        segment = slow_oscillation[start:end]
        peak = np.max(segment)
        trough = np.min(segment)
        trough_to_peak = np.append(trough_to_peak, peak - trough)
        trough_amplitude = np.append(trough_amplitude, np.abs(trough))

    return np.logical_and(
        trough_to_peak > np.mean(trough_to_peak) + np.std(trough_to_peak),
        trough_amplitude > np.mean(trough_amplitude) + np.std(trough_amplitude),
    )


def length_check(event_starts: np.ndarray, sampling_rate_lfp: float) -> np.ndarray:

    diffs = np.diff(event_starts) / sampling_rate_lfp
    valid = np.logical_and(0.8 < diffs, diffs < 2.0)
    # The final one is not valid as it doesn't go back up again
    valid = np.append(valid, False)
    return valid


def threshold_detect_falling_edge(signal: np.ndarray, threshold: float) -> np.ndarray:
    # Falling edge: From above to below the threshold
    falling_edges = (signal[:-1] > threshold) & (signal[1:] <= threshold)

    falling_indices = (
        np.where(falling_edges)[0] + 1
    )  # Shift by 1 to get the index where the crossing occurs

    return falling_indices
