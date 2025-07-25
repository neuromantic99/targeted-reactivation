from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from data_import import Session
from scipy.stats import zscore
from main import process_session
from plotting import plot_stim_triggered_average
from utils import get_aligners, get_data_paths

from ripples.utils_npyx import load_sync_npyx
from ripples.utils import threshold_detect


def load_spiking_data(
    kilosort_path: Path,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    spike_times = np.load(kilosort_path / "spike_times.npy")
    spike_clusters = np.load(kilosort_path / "spike_clusters.npy")
    label = pd.read_csv(
        kilosort_path / "cluster_group.tsv", sep="\t"
    ).KSLabel.to_numpy()

    # Probably don't need the cluster_id as it should be in order and 0 indexed but do it to make sure.
    cluster_id = pd.read_csv(
        kilosort_path / "cluster_group.tsv", sep="\t"
    ).cluster_id.to_numpy()
    # assert cluster id is sorted ascending
    assert np.all(np.diff(cluster_id) == 1)
    assert len(cluster_id) == len(label)
    id_to_label = dict(zip(cluster_id, label))
    label_array = np.array([id_to_label[cluster] for cluster in spike_clusters])

    return spike_times, spike_clusters, label_array


def main(data_path: Path, kilosort_path: Path) -> None:
    _, _, pycontrol_files = get_data_paths(data_path)
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    rsync_times = [session.times["rsync"] for session in sessions]

    if (kilosort_path.parent / "high_pass_sync.npy").exists():
        print("Loading existing sync file")
        npx_sync_times = np.load(kilosort_path.parent / "high_pass_sync.npy")
    else:
        sync = load_sync_npyx(kilosort_path.parent, filt_key="highpass")
        npx_sync_times = threshold_detect(sync, 0.5)
        np.save(kilosort_path.parent / "high_pass_sync.npy", npx_sync_times)

    assert sum(len(rsync) for rsync in rsync_times) == len(npx_sync_times)
    aligners = get_aligners(npx_sync_times, rsync_times)
    spike_times, spike_clusters, labels = load_spiking_data(kilosort_path)

    good = spike_times[labels == "good"]
    spike_clusters = spike_clusters[labels == "good"]

    sounds, LEDs = process_session(sessions[0])

    assert LEDs is not None

    sounds_times = aligners[0].B_to_A(
        np.array([sound.time for sound in sounds]), extrapolate=False
    )
    LEDs_times = aligners[0].B_to_A(
        np.array([led.time for led in LEDs]), extrapolate=False
    )
    plot_stim_triggered_average(
        LEDs_times, np.array([led.color for led in LEDs]), good, spike_clusters
    )
    plot_stim_triggered_average(
        sounds_times,
        np.array([sound.frequency for sound in sounds]),
        good,
        spike_clusters,
    )

    plt.show()


if __name__ == "__main__":
    data_path = Path("/Volumes/MarcBusche/Alex/Reactivations/2025-06-19/00055/")
    kilosort_path = data_path / "20250619_g0" / "20250619_g0_imec0" / "kilosort4"
    main(data_path, kilosort_path)
