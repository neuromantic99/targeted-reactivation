import json
from pathlib import Path
from typing import Literal
import numpy as np

from ripples.ripple_detection import (
    get_candidate_ripples,
    remove_duplicate_ripples,
    get_quality_metrics,
)

from ripples.utils import (
    compute_power,
    bandpass_filter,
)

from ripples.models import CandidateEvent

from consts import DETECTION_METHOD, RIPPLE_BAND
from models import RipplesCache
from utils import get_lfp_index_sleep_state

HERE = Path(__file__).parent


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


def detect_ripples(
    mouse: str,
    imec: str,
    ca1_low: int,
    ca1_high: int,
    data_folder: Path,
    sampling_rate_lfp: float,
    lfp: np.ndarray,
    session_type: Literal["conditioning", "resting", "tones"],
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
        lfp_det_chans=common_average_referenced,
        detection_channels_ca1=detection_channels_ca1,
        resting_ind=resting_ind,
        resting_ind_strict=resting_ind,
        sampling_rate=sampling_rate_lfp,
        detection_method=DETECTION_METHOD,
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
            candidate_events=ripples,
            lfp=common_average_referenced,
            common_average=common_average,
            sampling_rate=sampling_rate_lfp,
            detection_channels_ca1=detection_channels_ca1,
        )
    )

    if session_type != "conditioning":
        lfp_state_idx = get_lfp_index_sleep_state(
            data_folder=data_folder,
            # This mouse was done with buffering rather than streaming.
            # Need to remove ripples past the 30 minute mark below
            n_samples=lfp.shape[1] if mouse != "00053" else 30 * 60 * sampling_rate_lfp,
            sampling_rate_lfp=sampling_rate_lfp,
        )
        ripple_state = []
        for ripple in ripples:
            for state, idxs in lfp_state_idx.items():
                if ripple.peak_idx in idxs:
                    ripple_state.append(state)
                    break
    else:
        ripple_state = ["awake"] * len(ripples)

    assert len(ripple_state) == len(ripples)

    if mouse == "00053":
        ripples = [
            ripple
            for ripple in ripples
            if ripple.peak_idx < 30 * 60 * sampling_rate_lfp
        ]

    total_passing_all = np.sum(
        np.array(freq_check) & np.array(CAR_check_lr) & np.array(SRP_check_lr)
    )

    cache_result = RipplesCache(
        candidate_events=ripples,
        common_average_reference_check=CAR_check,
        common_average_reference_check_less_restrictive=CAR_check_lr,
        frequency_check=freq_check,
        super_ripple_check=SRP_check,
        super_ripple_check_less_restrictive=SRP_check_lr,
        length_recording=(
            lfp.shape[1] / sampling_rate_lfp if mouse != "00053" else 30 * 60
        ),
        state=ripple_state,
        state_lengths=(
            {state: len(list(idxs)) for state, idxs in lfp_state_idx.items()}
            if session_type != "conditioning"
            else {"awake": lfp.shape[1] / sampling_rate_lfp}
        ),
    )

    print(f"Total ripples passing all quality metrics: {total_passing_all}")
    print(f"Ripple rate {total_passing_all / (lfp.shape[1] / sampling_rate_lfp)}")
    with open(
        HERE / "results" / "ripples" / f"{mouse}_{imec}_{session_type}.json",
        "w",
    ) as f:
        json.dump(cache_result.model_dump(), f)
