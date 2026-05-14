from pathlib import Path

import numpy as np
import pandas as pd
from consts import LOCAL_SSD
from data_import import Session
from reactivation_classifier import get_sleep_state, process_probe
from utils import (
    build_path_dict,
    get_data_paths,
    process_session,
)

from gsheets_importer import gsheet2df
from lfp_signatures import get_ca1_rsc_channels

HERE = Path(__file__).parent


def process_mouse(
    data_path: Path, kilosort_paths: list[Path], paths_df: pd.DataFrame
) -> None:

    subject = data_path.parts[-1]
    sleep_state = get_sleep_state(data_path)
    for kilosort_path in kilosort_paths:

        imec_idx = int(kilosort_path.parent.parts[-1].split("imec")[1])

        ca1_low, ca1_high, rsc_low, rsc_high = get_ca1_rsc_channels(
            kilosort_path, paths_df
        )

        print(f"CA1: {ca1_low} - {ca1_high}, RSC: {rsc_low} - {rsc_high}")

        spike_times, spike_clusters, labels, closest_channel, aligners = process_probe(
            data_path,
            kilosort_path,
            (ca1_low, ca1_high, rsc_low, rsc_high),
            bin_data=False,
        )

        _, _, pycontrol_files = get_data_paths(data_path)
        sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]
        get_file_path = (
            lambda x: LOCAL_SSD / "for_zt" / subject / f"probe_{imec_idx}" / f"{x}.npy"
        )
        # create directory if it doesn't exist
        get_file_path("bla").parent.mkdir(parents=True, exist_ok=True)

        np.save(get_file_path("spike_times"), spike_times)
        np.save(get_file_path("spike_clusters"), spike_clusters)
        np.save(get_file_path("labels"), labels)
        np.save(get_file_path("closest_channel"), closest_channel)

        session_names = ["Conditioning", "Resting", "Targeted_reactivation"]

        region_dict = {
            "CA1": (ca1_low, ca1_high),
            "RSC": (rsc_low, rsc_high),
        }
        np.save(get_file_path("region_dict"), region_dict)

        for session_idx, session in enumerate(sessions):
            name = session_names[session_idx]

            sounds, LEDs = process_session(session)
            if session_idx == 0:
                LED_times_spikes = aligners[session_idx].B_to_A(
                    np.array([led.time for led in LEDs]), extrapolate=True
                )
                frequencies = np.array([sound.frequency for sound in sounds])

                sound_times_spikes = aligners[session_idx].B_to_A(
                    np.array([sound.time for sound in sounds]), extrapolate=True
                )
                colors = np.array([led.color for led in LEDs])
                np.save(
                    get_file_path(f"{name}_sound_times"),
                    sound_times_spikes,
                )
                np.save(get_file_path(f"{name}_sound_frequencies"), frequencies)
                np.save(
                    get_file_path(f"{name}_LED_times"),
                    LED_times_spikes,
                )
                np.save(get_file_path(f"{name}_LED_colors"), colors)
            elif session_idx == 2:
                frequencies = np.array([sound.frequency for sound in sounds])
                sound_times_spikes = aligners[session_idx].B_to_A(
                    np.array([sound.time for sound in sounds]), extrapolate=True
                )
                np.save(
                    get_file_path(f"{name}_sound_times"),
                    sound_times_spikes,
                )
                np.save(get_file_path(f"{name}_sound_frequencies"), frequencies)

    mouse_folder = get_file_path("bla").parent.parent
    np.save(mouse_folder / "sleep_state.npy", sleep_state)


def count_states(arr: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))


def main() -> None:

    path_dict = build_path_dict()
    paths_df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)

    sleep_state_dict = {}

    for mouse, kilosort_paths in path_dict.items():
        assert all(
            [
                (kilosort_path / "spike_times.npy").exists()
                for kilosort_path in kilosort_paths
            ]
        )
        data_path = kilosort_paths[0].parent.parent.parent

        process_mouse(data_path, kilosort_paths, paths_df)

        subject = data_path.parts[-1]
        sleep_state = get_sleep_state(data_path)
        sleep_state_dict[subject] = count_states(sleep_state)

    1 / 0


if __name__ == "__main__":
    main()
