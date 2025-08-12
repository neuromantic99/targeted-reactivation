import json
from pathlib import Path
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from ripples.utils import (
    compute_power,
    bandpass_filter,
    threshold_detect,
    compute_envelope,
    get_event_frequency,
)

from models import CandidateSpindle, SpindleCache
from utils import get_lfp_index_sleep_state


HERE = Path(__file__).parent


def detect_spindles(
    mouse: str,
    imec: str,
    rsc_low: int,
    rsc_high: int,
    data_folder: Path,
    sampling_rate_lfp: float,
    lfp: np.ndarray,
) -> None:
    # The mean across all channels was subtracted (common average reference) and the
    # mean across all times was subtracted from each channel to normalise
    lfp = lfp[rsc_low:rsc_high, :]
    common_average_time = np.mean(lfp, axis=0)
    lfp = np.subtract(lfp, common_average_time)
    common_average_channel = np.mean(lfp, axis=1)
    lfp = np.subtract(lfp, common_average_channel[:, np.newaxis])

    # After bandpass filtering the data from one cortical region at a time in the sigma band (10-16Hz)
    # using a zero-phase, second order Butterworth filter, the channel with the highest power in the sigma
    # band was selected for spindle detection.
    sigma_filtered = bandpass_filter(lfp, 10, 16, sampling_rate_lfp, order=2)
    max_power_channel = np.argmax(compute_power(sigma_filtered))

    # The filtered signal from this channel then underwent a Hilbert transform, and the absolute value of this was
    # taken as a proxy for instantaneous amplitude, or power. Candidate spindle events were identified where the
    # signal exceeded +2 standard deviations above the mean, and the start and end of these events was defined as
    # when the signal decayed below +0.2 standard deviations above the mean.
    envelope = compute_envelope(sigma_filtered)

    candidate_spindles = detect_candidate_spindles(
        envelope[max_power_channel, :], sampling_rate_lfp
    )
    control_band = bandpass_filter(lfp, 20, 30, sampling_rate_lfp, order=2)

    print(f"Number of candidate spindles: {len(candidate_spindles)}")

    # Candidate events were excluded if the instantaneous amplitude in a control frequency band of 20-30Hz exceeded
    # +4.5 standard deviations above the mean, in order to discount broadband power increases.

    control_band_envelope = compute_envelope(control_band)

    candidate_spindles = control_band_filter(
        candidate_spindles, control_band_envelope[max_power_channel, :]
    )

    print(
        f"Number of candidate spindles after control band filter: {len(candidate_spindles)}"
    )

    # Finally, the putative spindle events were required to be between 0.5 – 2.5 seconds in duration.
    #  In order to avoid duplicated detections where the signal oscillated over the detection threshold within a single event,
    #  if multiple events had the same start and end times, the version with the highest amplitude peak was kept, and the duplicates discarded.
    spindles = length_check(candidate_spindles, sampling_rate_lfp)

    print(f"Number of candidate spindles after length check: {len(spindles)}")

    lfp_state_idx = get_lfp_index_sleep_state(
        data_folder=data_folder,
        # This mouse was done with buffering rather than streaming.
        # Need to remove ripples past the 30 minute mark below
        n_samples=lfp.shape[1] if mouse != "00053" else 30 * 60 * sampling_rate_lfp,
        sampling_rate_lfp=sampling_rate_lfp,
    )

    if mouse == "00053":
        spindles = [s for s in spindles if s.peak_idx < 30 * 60 * sampling_rate_lfp]

    spindle_state = []

    for spindle in spindles:
        for state, idxs in lfp_state_idx.items():
            if spindle.peak_idx in idxs:
                spindle_state.append(state)
                break

    assert len(spindle_state) == len(spindles)

    cache_result = SpindleCache(
        spindles=spindles,
        length_recording=(
            lfp.shape[1] / sampling_rate_lfp if mouse != "00053" else 30 * 60
        ),
        state=spindle_state,
        state_lengths={state: len(list(idxs)) for state, idxs in lfp_state_idx.items()},
    )

    with open(
        HERE / "results" / "spindles" / f"{mouse}_{imec}.json",
        "w",
    ) as f:
        json.dump(cache_result.model_dump(), f)


def length_check(
    candidate_spindles: List[CandidateSpindle], sampling_rate: float
) -> List[CandidateSpindle]:
    filtered_spindles = []
    for spindle in candidate_spindles:
        duration = (spindle.offset - spindle.onset) / sampling_rate
        if 0.5 <= duration <= 2.5:
            filtered_spindles.append(spindle)

    return filtered_spindles


def control_band_filter(
    candidate_spindles: List[CandidateSpindle], control_band: np.ndarray
) -> List[CandidateSpindle]:
    filtered_spindles = []
    threshold = np.mean(control_band) + 4.5 * np.std(control_band)
    for spindle in candidate_spindles:
        control_band_segment = control_band[spindle.onset : spindle.offset]
        if np.max(control_band_segment) < threshold:
            filtered_spindles.append(spindle)
    return filtered_spindles


def detect_candidate_spindles(
    sigma_filtered: np.ndarray,
    sampling_rate: float,
) -> List[CandidateSpindle]:

    candidate_spindles: List[CandidateSpindle] = []
    upper_threshold = np.mean(sigma_filtered) + 2 * np.std(sigma_filtered)
    lower_threshold = np.mean(sigma_filtered) + 0.2 * np.std(sigma_filtered)

    in_event = False
    upper_exceeded = False
    peak_amp = -np.inf

    for idx, value in enumerate(sigma_filtered):

        if value > lower_threshold and not in_event:
            start_event = idx
            in_event = True

        if in_event and value > peak_amp:
            peak_amp = value
            peak_idx = idx

        # If you bounce on the lower threshold
        if value < lower_threshold and in_event and not upper_exceeded:
            in_event = False

        if value > upper_threshold:
            upper_exceeded = True

        if value < lower_threshold and in_event and upper_exceeded:
            in_event = False
            upper_exceeded = False
            # plt.axvline(start_event / 10, color="b", linestyle="--")
            # plt.axvline(idx / 10, color="b", linestyle="--")

            candidate_spindles.append(
                CandidateSpindle(
                    onset=start_event,
                    offset=idx,
                    peak_amplitude=peak_amp,
                    peak_idx=peak_idx,
                    frequency=get_event_frequency(
                        sigma_filtered[start_event:idx], sampling_rate
                    ),
                )
            )
            peak_amp = -np.inf

    return candidate_spindles
