from pathlib import Path, PureWindowsPath
from typing import Tuple
import numpy as np
import pandas as pd
from ripples.utils import compute_power, bandpass_filter, threshold_detect

from npyx import extract_rawChunk, read_metadata

from ripples.utils_npyx import load_sync_npyx
from ripples.ripple_detection import (
    get_candidate_ripples,
    remove_duplicate_ripples,
    get_quality_metrics,
)
from consts import LOCAL_SSD
from gsheets_importer import gsheet2df
from plotting import plot_lfp_profile, plot_lfp_spectrogram

from data_import import Session
from main import get_aligners

# from reactivation_classifier import get_ca1_rsc_mapping
from utils import get_data_paths
import matplotlib.pyplot as plt


def get_detection_channels_CA1(
    ca1: np.ndarray, ca1_low: int, ca1_high: int
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
    ca1_low, ca1_high, rsc_low, rsc_high = region_channels
    imec = f"imec_{str(lfp_path).split('imec')[1]}"
    data_folder = lfp_path.parent.parent

    _, _, pycontrol_files = get_data_paths(data_folder)
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    assert len(sessions) == 3

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

    start_rest = aligners[1].first_matched_time_A / sampling_rate_lfp
    end_rest = aligners[1].last_matched_time_A / sampling_rate_lfp

    resting_lfp_path = LOCAL_SSD / f"resting_lfps/{mouse}_{imec}.npy"
    if (resting_lfp_path).exists():
        print("loading existing resting LFP")
        lfp = np.load(resting_lfp_path)
    else:
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
    #     ripple_band=[120, 250],
    #     sampling_rate_lfp=sampling_rate_lfp,
    #     mouse=mouse,
    #     imec=imec,
    # )

    ca1_lfp = lfp[ca1_low : ca1_high + 1, :]

    DETECTION_METHOD = "median"  # options 'median' or 'sd'
    RIPPLE_BAND = [120, 250]
    SUPRA_RIPPLE_BAND = [250, 500]

    detection_channels_ca1 = get_detection_channels_CA1(ca1_lfp, ca1_low, ca1_high)

    assert len(detection_channels_ca1) == 5, "Expected 5 detection channels in CA1"
    assert min(detection_channels_ca1) >= ca1_low
    assert max(detection_channels_ca1) < ca1_high

    ca1_lfp_detection_channels_only = lfp[detection_channels_ca1, :]
    common_average = np.nanmedian(ca1_lfp, axis=0)
    common_average_referenced = np.subtract(
        ca1_lfp_detection_channels_only, common_average
    )
    assert common_average_referenced.shape[0] == len(detection_channels_ca1)

    # TODO: this can be used to filter for nrem
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
    total_passing_all = np.sum(
        np.array(freq_check) & np.array(CAR_check_lr) & np.array(SRP_check_lr)
    )

    print(f"Total ripples passing all quality metrics: {total_passing_all}")
    print(f"Ripple rate {total_passing_all / (lfp.shape[1] / sampling_rate_lfp)}")
    plt.show()


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
                row["CA1_Low"],
                row["CA1_High"],
                row["RSC_Low"],
                row["RSC_High"],
            )
    raise ValueError(f"Could not find channels for {lfp_file.name} in dataframe")


def main() -> None:
    df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)

    umbrella = Path("/Volumes/MarcBusche/Alex/Reactivations/")

    lfp_files = list(umbrella.rglob("*.lf.bin"))
    assert len(lfp_files) > 0, "No LFP files found"

    for lfp_file in lfp_files:
        ca1_low, ca1_high, rsc_low, rsc_high = get_ca1_rsc_channels(lfp_file, df)

        assert ca1_low < ca1_high < rsc_low < rsc_high
        assert 30 < ca1_high - ca1_low < 70
        assert 80 < rsc_high - rsc_low < 120

        get_ripple_rate(
            lfp_path=lfp_file.parent,
            region_channels=(ca1_low, ca1_high, rsc_low, rsc_high),
        )

    # get_ripple_rate(
    #     Path(
    #         "/Volumes/MarcBusche/Alex/Reactivations/2025-06-09/00052/20250609_g1/20250609_g1_imec0"
    #     )
    #     # Path(
    #     #     "/Volumes/MarcBusche/Alex/Reactivations/2025-06-10/10682/20250610_g0/20250610_g0_imec0"
    #     # )
    # )


if __name__ == "__main__":
    main()
