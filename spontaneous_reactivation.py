import sys
from pathlib import Path
from typing import Tuple
import statsmodels.formula.api as smf

from matplotlib import pyplot as plt
import numpy as np
from opt_einsum import contract
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import seaborn as sns

from consts import LOCAL_SSD, SERVER_CACHE_PATH
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

    offline_activity_matrix = zscore(offline_activity_matrix, axis=1)
    # Remove nans from silent neurons
    offline_activity_matrix = np.nan_to_num(offline_activity_matrix)

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

    # Since the sign of the output of ICA is arbitrary, the signs of the weight vector were set such that
    # the highest absolute weight was set to positive.
    # pmc.ncbi.nlm.nih.gov/articles/instance/10760112/bin/NIHPP2023.12.12.571373V1-supplement-1.pdf
    for b in range(n_components):
        max_idx = np.argmax(np.abs(ensemble_matrix[:, b]))
        if ensemble_matrix[max_idx, b] < 0:
            ensemble_matrix[:, b] *= -1  # flip entire column

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

    all_components = []
    mouse_id = []

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

        redo = True

        for kilosort_path in kilosort_paths:
            region_boundaries = get_ca1_rsc_channels(kilosort_path, paths_df)
            imec = kilosort_path.parent.name.split("_")[-1]
            if (SERVER_CACHE_PATH / f"{mouse}_{imec}.npy").exists() and not redo:
                reactivation_strength = np.load(
                    SERVER_CACHE_PATH / f"{mouse}_{imec}.npy"
                )
                resting_bin_edges = np.load(
                    SERVER_CACHE_PATH / f"{mouse}_{imec}_binedges.npy"
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
                    SERVER_CACHE_PATH / f"{mouse}_{imec}",
                    reactivation_strength,
                )

                np.save(
                    SERVER_CACHE_PATH / f"{mouse}_{imec}_binedges",
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
            # Add a small buffer to ensure the last ripple is included
            assert resting_bin_edges[-1] > np.max(ripple_times_spikes) - 300

            bin_size = resting_bin_edges[1] - resting_bin_edges[0]
            trial_size = (30_000 * 8) // bin_size
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

            print(
                f"{mouse} {imec} number nrem trials: {n_rem_trials.shape[0]}. number awake trials: {awake_trials.shape[0]}"
            )

            components = []
            for component in range(n_rem_trials.shape[1]):
                assembly_data = n_rem_trials[:, component, :]
                # compute global mean/std across ripples and time
                mean = assembly_data.mean()
                std = assembly_data.std(ddof=1)
                # zscore the whole (n_ripples, n_time) block
                zdata = (assembly_data - mean) / std
                # average across ripples -> one trace per assembly
                components.append(zdata.mean(axis=0))

            all_components.extend(components)
            mouse_id.extend([mouse] * len(components))

            x_axis = np.arange(-trial_size, trial_size) * bin_size / 30_000

    peak_start = np.where(x_axis == -0.2)[0][0]
    peak_end = np.where(x_axis == 0.2)[0][0]

    df = pd.DataFrame(
        {
            "result": [np.mean(comp[peak_start:peak_end]) for comp in all_components],
            "mouse": mouse_id,
            "genotype": ["WT" if m[:3] == "000" else "NLGF" for m in mouse_id],
        }
    )

    model = smf.mixedlm(
        "result ~ genotype",
        df,
        groups=df["mouse"],
        use_sqrt=True,
    )
    model_fit = model.fit(reml=False)
    assert model_fit.converged
    print(model_fit.summary())
    wt_results = np.array(
        [comp for idx, comp in enumerate(all_components) if mouse_id[idx][:3] == "000"]
    )
    nlgf_results = np.array(
        [comp for idx, comp in enumerate(all_components) if mouse_id[idx][:3] != "000"]
    )

    plt.figure()
    shaded_line_plot(wt_results, x_axis=x_axis, color="blue", label="WT")
    shaded_line_plot(nlgf_results, x_axis=x_axis, color="red", label="NLGF")

    plt.figure()
    sns.boxplot(data=df, x="genotype", y="result")

    plt.figure()
    for comp in wt_results:

        plt.plot(
            x_axis,
            comp,
            color="blue",
            alpha=0.2,
        )

    for comp in nlgf_results:
        plt.plot(
            x_axis,
            comp,
            color="red",
            alpha=0.2,
        )

    1 / 0


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

    start_conditioning, end_conditioning = aligners[0].B_to_A(
        np.array(
            [pycontrol_conditioning_time_edges[0], pycontrol_conditioning_time_edges[1]]
        )
    )

    bin_width = 0.02
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
        matrix.append(binned)

    return np.array(matrix), bin_edges


if __name__ == "__main__":
    main()
