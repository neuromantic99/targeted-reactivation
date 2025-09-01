import sys
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from opt_einsum import contract
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

from consts import LOCAL_SSD
from data_import import Session
from gsheets_importer import gsheet2df
from lfp_signatures import get_ca1_rsc_channels
from models import RipplesCache
from plotting import shaded_line_plot
from reactivation_classifier import get_training_data, process_probe
from rsync import Rsync_aligner
from utils import build_path_dict, get_aligners, get_data_paths, shuffle_rows

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


def fast_ica_sklearn(X: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.decomposition import FastICA

    # Don't need to do this
    # X_centered = X - np.mean(X, axis=1, keepdims=True)

    X_centered = X

    # Transpose to (n_samples, n_features) for sklearn
    X_centered = X_centered.T

    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        fun="logcosh",  # logcosh with alpha = 1 is the same as tanh used in matlab
        fun_args={"alpha": 1.0},
        max_iter=500,
        random_state=0,  # Optional: for reproducibility
    )
    S = ica.fit_transform(X_centered)  # Shape: (n_timepoints, n_components)
    W = ica.components_  # Shape: (n_components, n_cells)

    return W.T


def compute_ICA_components(ssp_vectors: np.ndarray) -> np.ndarray:
    # 'the elements of M (in our case σ2 = 1 due to z-score normalization), Ncolumns is the number of columns and Nrows the number of rows.'
    n_rows, n_cols = ssp_vectors.shape

    # 'spiking matrix' (Lopes-dos-Santos):
    # (neurons, time) ('each matrix entry denotes the number of spikes of a given neuron (rows) in a given time bin (columns)')
    # ssp_vectors is (n_place_cells, n_frames)

    # 'Next, the spike count of each neuron (i.e., each row of the matrix) is normalized by z-score transformation'
    ssp_vectors_z = zscore(ssp_vectors, axis=1)

    # 'in our case the covariance matrix is equal to the correlation matrix, and can be calculated as:
    # C = Z*Z.T / Ncolumns
    # where Z is the (z-scored) spike matrix, T the transpose operator, and Ncolumns is the number of time bins of Z.
    # Thus, the element at the i-th column and j-th row of C is the linear correlation between neurons i and j.'
    covariance_matrix = (ssp_vectors_z @ ssp_vectors_z.T) / n_cols

    # 'Since C is necessarily real and symmetric, it follows from the spectral theorem that it can be decomposed'
    # 'Compute the eigenvalues and right eigenvectors of a square array.' (NumPy documentation)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 'where σ2 is the variance of the elements of M (in our case σ2 = 1 due to z-score normalization)'
    assert np.isclose(np.var(ssp_vectors_z), 1)

    q = n_cols / n_rows

    # 'with q = Ncolumns/Nrows ≥ 1'
    assert q >= 1

    # 'λmax and λmin are the maximum and minimum bounds, respectively, and are calculated as:'
    lambda_max = (1 + np.sqrt(1 / q)) ** 2

    # 'Thus, if the rows of M are statistically independent, the probability of finding an eigenvalue outside these bounds is zero.
    #  In other words, the variance of the data in any axis cannot be larger than λmax when neurons are uncorrelated.
    #  Therefore, λmax can be used as a statistical threshold for detecting cell assembly activity'
    n_significant_components = np.sum(eigenvalues > lambda_max)
    print(f"Number of significant components: {n_significant_components}")

    if n_significant_components < 1:
        raise ValueError("No significant components found")

    return fast_ica_sklearn(ssp_vectors_z, n_significant_components)


def offline_reactivation(
    offline_activity_matrix: np.ndarray,
    ensemble_matrix: np.ndarray,
    do_shuffle: bool = False,
) -> np.ndarray:
    """
    For each component, b, of ICA ensemble matrix w, a
    square projection matrix, P, was computed from wb as follows:
    Pb = wb * wbT
    Where T denotes the transpose operator. Subsequently, the diagonal of the
    projection matrix P was set to zero to exclude each cell's individual firing rate
    variance.
    Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z.
    For the ith time point (frame) in Z, the reactivation strength Rb,i
    of the bth ICA component was calculated as the square of the projection length of Zi on Pb as follows:
    Rbi = ZiT * Pb * Zb
    """

    if do_shuffle:
        # """ICA components were shuffled by randomly permuting the weight matrix w across
        # PCs and recalculating the reactivation strength."""
        # Confusing, should we shuffle rows or columns (i.e. PCs or components)?
        # ensemble_matrix = ensemble_matrix[
        #     :, np.random.permutation(ensemble_matrix.shape[1])
        # ]
        ensemble_matrix = shuffle_rows(ensemble_matrix)

    n_timepoints = offline_activity_matrix.shape[1]
    n_cells = ensemble_matrix.shape[0]
    n_components = ensemble_matrix.shape[1]

    # Einstein summation convention
    # components -> b
    # frames -> i
    # cells -> k
    # j (place holder)
    # P -> (components, cells, cells) -> P[b, j, k]
    # w -> (cells, components) -> w[b, j]
    # Z -> (cells, timepoints) -> Z[c, i]
    # R -> (components, timepoints) -> R[b, i]

    # outer product for each component b with itself
    # shape (components, cells, cells)
    projection_matrices = contract("kb,jb->bkj", ensemble_matrix, ensemble_matrix)
    assert projection_matrices.shape == (
        n_components,
        n_cells,
        n_cells,
    )

    # set diagonal to zero
    for b in range(projection_matrices.shape[0]):
        np.fill_diagonal(projection_matrices[b], 0)

    reactivation_strength = contract(
        "ik,bkj,ji->bi",
        offline_activity_matrix.T,  # Zi.T
        projection_matrices,  # Pb
        offline_activity_matrix,  # Zi
    )

    assert reactivation_strength.shape == (
        n_components,
        n_timepoints,
    )

    return reactivation_strength


def main() -> None:

    path_dict = build_path_dict()
    paths_df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)

    for mouse, kilosort_paths in path_dict.items():
        assert all(
            [
                (kilosort_path / "spike_times.npy").exists()
                for kilosort_path in kilosort_paths
            ]
        )

        data_path = kilosort_paths[0].parent.parent.parent
        _, frame_trigger_times, pycontrol_files = get_data_paths(data_path)
        frame_triggers = np.load(frame_trigger_times[1])
        start_time_pycontrol, end_time_pycontrol = frame_triggers[0], frame_triggers[-1]
        sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

        redo = True

        for kilosort_path in kilosort_paths:
            region_boundaries = get_ca1_rsc_channels(kilosort_path, paths_df)
            imec = kilosort_path.parent.name.split("_")[-1]
            if (
                HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}.npy"
            ).exists() and not redo:
                reactivation_strength = np.load(
                    HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}.npy"
                )
                bin_edges = np.load(
                    HERE
                    / "results"
                    / "reactivation_strength"
                    / f"{mouse}_{imec}_binedges.npy"
                )
            else:
                reactivation_strength, bin_edges = get_reactivation_strength(
                    mouse=mouse,
                    imec=imec,
                    data_path=data_path,
                    sessions=sessions,
                    kilosort_path=kilosort_path,
                    region_boundaries=region_boundaries,
                    start_time_pycontrol=start_time_pycontrol,
                    end_time_pycontrol=end_time_pycontrol,
                )

            with open(
                HERE / "results" / "ripples" / f"{mouse}_imec_{imec[-1]}.json"
            ) as f:
                ripple_cache = RipplesCache.model_validate_json(f.read())

            passing_checks = (
                np.array(ripple_cache.common_average_reference_check)
                & np.array(ripple_cache.frequency_check)
                & np.array(ripple_cache.super_ripple_check)
            )
            if mouse == "00053":
                passing_checks = passing_checks[: len(ripple_cache.candidate_events)]
            ripples = np.array(ripple_cache.candidate_events)[passing_checks]
            state = np.array(ripple_cache.state)[passing_checks]

            # Assumes that you've already loaded the lfp syncs to the local ssd using lfp_signatures/get_sync()
            lfp_sync = np.load(
                Path(LOCAL_SSD / "lfp_syncs")
                / f"npx_sync_times_{mouse}_imec_{imec[-1]}.npy"
            )
            spikes_sync = np.load(kilosort_path.parent / "high_pass_sync.npy")
            assert len(lfp_sync) == len(spikes_sync)

            lfp_spikes_aligner = Rsync_aligner(
                spikes_sync,
                lfp_sync,
                raise_exception=True,
            )
            assert int(lfp_spikes_aligner.units_B) == 30_000 / 2500

            # The ripple times are relative to the first sample in the resting session, so find this sample
            # relative to the whole recording
            lfp_pycontrol_aligner = get_aligners(
                lfp_sync, [session.times["rsync"] for session in sessions]
            )[1]
            first_sample = lfp_pycontrol_aligner.B_to_A(
                np.array([start_time_pycontrol])
            )[0]

            ripple_times_spikes = np.array(
                [
                    lfp_spikes_aligner.B_to_A(
                        np.array([r.onset + first_sample, r.offset + first_sample])
                    ).astype(int)
                    for r in ripples
                ]
            )
            assert bin_edges[0] < np.min(ripple_times_spikes)
            assert bin_edges[-1] > np.max(ripple_times_spikes)
            # plt.plot(bin_edges[:-1], np.mean(reactivation_strength, axis=0))
            # for r in ripple_times_spikes:
            #     plt.axvspan(r[0], r[1], alpha=0.5, color="red")

            bin_size = bin_edges[1] - bin_edges[0]
            trial_size = 15_000 // bin_size  # 1 second either side
            n_rem_trials = []
            awake_trials = []

            for idx, r in enumerate(ripple_times_spikes):
                if state[idx] not in ["nrem", "awake"]:
                    continue

                closest_bin = np.argmin(np.abs(bin_edges - r[0]))
                trial_response = reactivation_strength[
                    :, closest_bin - trial_size : closest_bin + trial_size
                ]
                if state[idx] == "nrem":
                    n_rem_trials.append(trial_response)
                elif state[idx] == "awake":
                    awake_trials.append(trial_response)

            n_rem_trials = np.array(n_rem_trials)
            awake_trials = np.array(awake_trials)

            x_axis = np.arange(-trial_size, trial_size) * bin_size / 30_000

            shaded_line_plot(
                zscore(n_rem_trials.sum(1), axis=1), x_axis, "blue", "NREM"
            )

            shaded_line_plot(
                zscore(awake_trials.sum(1), axis=1), x_axis, "red", "awake"
            )

            1 / 0


def get_reactivation_strength(
    mouse: str,
    imec: str,
    data_path: Path,
    sessions: str,
    kilosort_path: str,
    region_boundaries: str,
    start_time_pycontrol: float,
    end_time_pycontrol: float,
):

    spike_times, spike_clusters, labels, closest_channel, aligners = process_probe(
        data_path=data_path,
        kilosort_path=kilosort_path,
        region_boundaries=region_boundaries,
        bin_data=False,
    )

    # 5 seconds either side of the LED in 1000 bins, so 100 ms
    window = 5
    n_bins = 1000

    bin_width = (2 * window) / n_bins

    train_array, y_train = get_training_data(
        sessions, aligners, spike_clusters, spike_times, n_bins, window
    )
    train_array = np.swapaxes(train_array, 0, 1)

    # Float cast to guard against smoothing ints rounding values
    ssp_vectors = train_array.reshape(train_array.shape[0], -1).astype(np.float32)

    # 100 ms sigma
    ssp_vectors = gaussian_filter1d(ssp_vectors, sigma=int(0.1 / bin_width), axis=1)

    # Sometimes get a cell that never spiked
    clusters_keep = np.sum(ssp_vectors, axis=1) > 0
    ssp_vectors = ssp_vectors[clusters_keep, :]
    comps = compute_ICA_components(ssp_vectors)

    # Done in the same way as the lfp
    start_rest, end_rest = aligners[1].B_to_A(
        np.array([start_time_pycontrol, end_time_pycontrol])
    )
    rest_spikes_idx = (spike_times >= start_rest) & (spike_times <= end_rest)
    rest_spikes = spike_times[rest_spikes_idx]
    rest_spike_clusters = spike_clusters[rest_spikes_idx]

    bin_edges = np.arange(int(start_rest), int(end_rest), int(bin_width * 30_000))
    reactivation = []

    for cluster_id in np.unique(spike_clusters)[clusters_keep]:
        spike_times_cluster = rest_spikes[rest_spike_clusters == cluster_id]
        binned = np.histogram(spike_times_cluster, bins=bin_edges)[0].astype(np.float32)
        reactivation.append(gaussian_filter1d(binned, sigma=int(0.1 / bin_width)))

    reactivation = np.array(reactivation)
    assert reactivation.shape[0] == ssp_vectors.shape[0]
    reactivation_strength = offline_reactivation(reactivation, comps, do_shuffle=False)
    print(f"reactivation strength found for {mouse} {imec}, saving")
    np.save(
        HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}",
        reactivation_strength,
    )

    np.save(
        HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}_binedges",
        bin_edges,
    )
    return reactivation_strength, bin_edges


if __name__ == "__main__":
    main()
