import sys
from pathlib import Path
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
from opt_einsum import contract
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

from consts import LOCAL_SSD
from data_import import Session
from gsheets_importer import gsheet2df
from lfp_signatures import get_ca1_rsc_channels
from models import RipplesCache
from plotting import shaded_line_plot
from reactivation_classifier import process_probe
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
        conditioning_frames = np.load(frame_trigger_times[0])
        pycontrol_conditioning_time_edges = (
            conditioning_frames[0],
            conditioning_frames[-1],
        )
        resting_frames = np.load(frame_trigger_times[1])
        pycontrol_resting_time_edges = (resting_frames[0], resting_frames[-1])

        sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

        redo = False

        for kilosort_path in kilosort_paths:
            region_boundaries = get_ca1_rsc_channels(kilosort_path, paths_df)
            imec = kilosort_path.parent.name.split("_")[-1]
            if (
                HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}.npy"
            ).exists() and not redo:
                reactivation_strength = np.load(
                    HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}.npy"
                )
                resting_bin_edges = np.load(
                    HERE
                    / "results"
                    / "reactivation_strength"
                    / f"{mouse}_{imec}_binedges.npy"
                )
            else:
                reactivation_strength, resting_bin_edges = get_reactivation_strength(
                    data_path=data_path,
                    kilosort_path=kilosort_path,
                    region_boundaries=region_boundaries,
                    pycontrol_conditioning_time_edges=pycontrol_conditioning_time_edges,
                    pycontrol_resting_time_edges=pycontrol_resting_time_edges,
                )

                print(f"reactivation strength found for {mouse} {imec}, saving")
                np.save(
                    HERE / "results" / "reactivation_strength" / f"{mouse}_{imec}",
                    reactivation_strength,
                )

                np.save(
                    HERE
                    / "results"
                    / "reactivation_strength"
                    / f"{mouse}_{imec}_binedges",
                    resting_bin_edges,
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
            # Some random sync pulses at the start from failed pycontrol sessions
            if mouse == "11150":
                spikes_sync = spikes_sync[-5435:]

            assert len(lfp_sync) == len(spikes_sync)

            lfp_spikes_aligner = Rsync_aligner(
                spikes_sync,
                lfp_sync,
                raise_exception=True,
            )
            assert round(lfp_spikes_aligner.units_B) == 30_000 / 2500

            # The ripple times are relative to the first sample in the resting session, so find this sample
            # relative to the whole recording
            lfp_pycontrol_aligner = get_aligners(
                lfp_sync, [session.times["rsync"] for session in sessions]
            )[1]
            first_sample = lfp_pycontrol_aligner.B_to_A(
                np.array([pycontrol_resting_time_edges[0]])
            )[0]

            ripple_times_spikes = np.array(
                [
                    lfp_spikes_aligner.B_to_A(
                        np.array(
                            [
                                r.onset + first_sample,
                                r.offset + first_sample,
                                r.peak_idx + first_sample,
                            ]
                        )
                    ).astype(int)
                    for r in ripples
                ]
            )
            assert resting_bin_edges[0] < np.min(ripple_times_spikes)
            assert resting_bin_edges[-1] > np.max(ripple_times_spikes)

            bin_size = resting_bin_edges[1] - resting_bin_edges[0]
            trial_size = 15_000 // bin_size  # 0.5 seconds either side
            n_rem_trials = []
            awake_trials = []

            for idx, r in enumerate(ripple_times_spikes):
                if state[idx] not in ["nrem", "awake"]:
                    continue
                closest_bin = np.argmin(np.abs(resting_bin_edges - r[2]))
                if (
                    closest_bin - trial_size < 0
                    or closest_bin + trial_size > resting_bin_edges.shape[0]
                ):
                    continue

                trial_response = reactivation_strength[
                    :, closest_bin - trial_size : closest_bin + trial_size
                ]

                assert trial_response.shape == (
                    reactivation_strength.shape[0],
                    2 * trial_size,
                )

                if state[idx] == "nrem":
                    n_rem_trials.append(trial_response)
                elif state[idx] == "awake":
                    awake_trials.append(trial_response)

            n_rem_trials = np.array(n_rem_trials)
            awake_trials = np.array(awake_trials)

            x_axis = np.arange(-trial_size, trial_size) * bin_size / 30_000

            def zscore_components(arr: np.ndarray) -> np.ndarray:
                reshaped = arr.reshape(arr.shape[0] * arr.shape[2], arr.shape[1])
                zscored = zscore(reshaped, axis=0)
                return zscored.reshape(arr.shape)

            plt.figure()

            # shaded_line_plot(
            #     zscore_components(n_rem_trials).mean(1), x_axis, "blue", "NREM"
            # )
            # shaded_line_plot(
            #     zscore_components(awake_trials).mean(1), x_axis, "red", "awake"
            # )
            n_rem_trials = zscore_components(n_rem_trials)
            awake_trials = zscore_components(awake_trials)

            for comp in range(n_rem_trials.shape[1]):
                plt.plot(
                    x_axis, n_rem_trials[:, comp, :].mean(0), color="blue", alpha=0.5
                )

            for comp in range(awake_trials.shape[1]):
                plt.plot(
                    x_axis, awake_trials[:, comp, :].mean(0), color="red", alpha=0.5
                )

            plt.savefig(
                HERE
                / "plots"
                / "reactivation_strength"
                / f"{mouse}_{imec}_reactivation_strength.png"
            )
            plt.title(f"{mouse}_{imec}_reactivation_strength")


def get_reactivation_strength(
    data_path: Path,
    kilosort_path: str,
    region_boundaries: str,
    pycontrol_conditioning_time_edges: Tuple[float, float],
    pycontrol_resting_time_edges: Tuple[float, float],
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

    start_conditioning, end_conditioning = aligners[0].B_to_A(
        np.array(
            [pycontrol_conditioning_time_edges[0], pycontrol_conditioning_time_edges[1]]
        )
    )

    ssp_vectors, _ = build_cluster_matrix(
        spike_times, spike_clusters, start_conditioning, end_conditioning, bin_width
    )

    # Sometimes get a cell that never spiked
    clusters_keep = np.sum(ssp_vectors, axis=1) > 0
    ssp_vectors = ssp_vectors[clusters_keep, :]
    comps = compute_ICA_components(ssp_vectors)

    # Done in the same way as the lfp
    start_rest, end_rest = aligners[1].B_to_A(
        np.array([pycontrol_resting_time_edges[0], pycontrol_resting_time_edges[1]])
    )

    reactivation, resting_bin_edges = build_cluster_matrix(
        spike_times, spike_clusters, start_rest, end_rest, bin_width
    )

    reactivation = reactivation[clusters_keep, :]

    assert reactivation.shape[0] == ssp_vectors.shape[0]
    reactivation_strength = offline_reactivation(reactivation, comps, do_shuffle=False)

    return reactivation_strength, resting_bin_edges


def build_cluster_matrix(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    start: float,
    end: float,
    bin_width: float,
) -> Tuple[np.ndarray, np.ndarray]:

    bin_edges = np.arange(int(start), int(end), int(bin_width * 30_000))

    spike_idxs = (spike_times >= start) & (spike_times <= end)
    spikes_matrix = spike_times[spike_idxs]
    clusters_matrix = spike_clusters[spike_idxs]

    matrix = []

    for cluster_id in np.unique(spike_clusters):
        spike_times_cluster = spikes_matrix[clusters_matrix == cluster_id]
        binned = np.histogram(spike_times_cluster, bins=bin_edges)[0].astype(np.float32)
        matrix.append(gaussian_filter1d(binned, sigma=int(0.1 / bin_width)))

    return np.array(matrix), bin_edges


if __name__ == "__main__":
    main()
