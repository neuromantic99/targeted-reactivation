import pickle
import sys
from pathlib import Path
from typing import List, Literal, Tuple
from sklearn.metrics import auc
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind

from matplotlib import pyplot as plt
import numpy as np
from opt_einsum import contract
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import seaborn as sns

from consts import LOCAL_SSD, RIPPLE_PATH, SERVER_CACHE_PATH
from data_import import Session
from gsheets_importer import gsheet2df
from lfp_signatures import get_ca1_rsc_channels
from models import RipplesCache, SpindleCache
from plotting import shaded_line_plot
from reactivation_classifier import process_probe
from rsync import Rsync_aligner
from utils import (
    build_path_dict,
    get_aligners,
    get_data_paths,
    shuffle_rows,
    zero_same_region,
)
from scipy.stats import ttest_rel
from ripples.models import CandidateEvent

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

sns.set_context("talk")


FIGURE_PATH = Path("/Volumes/MarcBusche/James/figures")


def compute_pcc_scores(
    offline_activity_matrix: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """
    To assess the xth cell's contribution to ICA reactivation, a PCC score was defined as the mean across all components b and
    timepoints i of the reactivation score R computed from all PCs c minus the reactivation score Rcx computed after
    the exclusion xth cell from the activity and template matrices.
    """
    # TODO: should we keep offline_reactivation for computing R_full separately?
    # TODO: or should we try and change offline_reactivation to use the approach below?
    # this function is bypassing the offline_reactivation function to be faster while returning close results
    # it has been tested that the results are close to the non-vectorised approach

    n_cells = offline_activity_matrix.shape[0]
    n_timepoints = offline_activity_matrix.shape[1]
    n_components = ensemble_matrix.shape[1]

    Z = offline_activity_matrix  # (n_cells, n_timepoints)
    w = ensemble_matrix  # (n_cells, n_components)

    # for each component b compute w_b @ Z
    wZ = contract("kb,ki->bi", w, Z)  # (n_components, n_timepoints)
    assert wZ.shape == (n_components, n_timepoints)

    # for each cell k and component b: compute contribution
    # contribution_kb = 2 * w[k,b] * Z[k,:] * (wZ[b,:] - w[k,b] * Z[k,:])
    contributions = np.zeros((n_cells, n_components, n_timepoints))

    for k in range(n_cells):
        # w[k,:] is (n_components,), Z[k,:] is (n_timepoints,)
        # wZ is (n_components, n_timepoints)
        w_k = w[k, :, np.newaxis]  # (n_components, 1)
        assert w_k.shape == (n_components, 1)
        Z_k = Z[k, np.newaxis, :]  # (1, n_timepoints)
        assert Z_k.shape == (1, n_timepoints)

        # other cells' contribution for each component and timepoint
        other_contrib = wZ - w_k * Z_k  # (n_components, n_timepoints)
        assert other_contrib.shape == (n_components, n_timepoints)

        # cell k's contribution
        contributions[k] = 2 * w_k * Z_k * other_contrib

    pcc_scores = np.mean(contributions, axis=(1, 2))
    assert pcc_scores.shape == (n_cells,)

    return pcc_scores


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
    assert np.isclose(np.var(ssp_vectors_z), 1, atol=1e-3)

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
    cluster_regions: np.ndarray | None = None,
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

    projection_matrices = (
        np.array(
            [zero_same_region(mat, cluster_regions) for mat in projection_matrices]
        )
        if cluster_regions is not None
        else projection_matrices
    )

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

    # This stores the data as in Shin et al.
    all_components = []
    # This stores each component x ripple x time for further inspection
    all_raw_components = []
    mouse_ids_components = []

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

        for kilosort_path in kilosort_paths:
            region_boundaries = get_ca1_rsc_channels(kilosort_path, paths_df)
            imec = kilosort_path.parent.name.split("_")[-1]

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
            redo = False

            if (SERVER_CACHE_PATH / f"{mouse}_{imec}.npy").exists() and not redo:
                reactivation_strength = np.load(
                    SERVER_CACHE_PATH / f"{mouse}_{imec}.npy"
                )
                resting_bin_edges = np.load(
                    SERVER_CACHE_PATH / f"{mouse}_{imec}_binedges.npy"
                )
            else:
                ripples, _ = load_ripples(mouse, imec, session_type="conditioning")
                ripple_times_spikes_conditioning = get_ripple_times_in_spikes(
                    spikes_sync=spikes_sync,
                    lfp_sync=lfp_sync,
                    pycontrol_time_edges=pycontrol_conditioning_time_edges,
                    sessions=sessions,
                    ripples=ripples,
                )

                reactivation_strength, resting_bin_edges, pcc_scores = (
                    get_reactivation_strength(
                        data_path=data_path,
                        kilosort_path=kilosort_path,
                        region_boundaries=region_boundaries,
                        pycontrol_conditioning_time_edges=pycontrol_conditioning_time_edges,
                        pycontrol_resting_time_edges=pycontrol_resting_time_edges,
                        ripple_times_spikes=ripple_times_spikes_conditioning,
                    )
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
                np.save(
                    SERVER_CACHE_PATH / f"{mouse}_{imec}_pcc_scores",
                    pcc_scores,
                )
                continue

            components, raw_components = align_reactivation_to_ripples(
                mouse=mouse,
                imec=imec,
                lfp_sync=lfp_sync,
                spikes_sync=spikes_sync,
                pycontrol_resting_time_edges=pycontrol_resting_time_edges,
                sessions=sessions,
                resting_bin_edges=resting_bin_edges,
                reactivation_strength=reactivation_strength,
                alignment_point="peak",
            )

            all_components.extend(components)
            all_raw_components.extend(raw_components)
            mouse_ids_components.extend([mouse] * len(components))

    with open("all_components.pkl", "wb") as f:
        pickle.dump(all_components, f)

    with open("all_raw_components.pkl", "wb") as f:
        pickle.dump(all_raw_components, f)

    np.save("all_mouse_ids.npy", np.array(mouse_ids_components))

    plot_component_traces(all_components, mouse_ids_components, resting_bin_edges)


def plot_component_traces(all_components, mouse_id, resting_bin_edges):

    nan_idx = np.any(np.isnan(all_components), axis=1)
    all_components = np.array(all_components)[~nan_idx, :]
    mouse_id = np.array(mouse_id)[~nan_idx]

    bin_size = resting_bin_edges[1] - resting_bin_edges[0]
    trial_size = (30_000 * 8) // bin_size
    x_axis = np.arange(-trial_size, trial_size) * bin_size / 30_000
    peak_start = np.where(x_axis == -0.2)[0][0]
    peak_end = np.where(x_axis == 0.2)[0][0]

    peak_start = np.where(x_axis == -0.2)[0][0]
    peak_end = np.where(x_axis == 0.2)[0][0]

    df = pd.DataFrame(
        {
            "result": [
                auc(x_axis[peak_start:peak_end], comp[peak_start:peak_end])
                for comp in all_components
            ],
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
    model_fit = model.fit(reml=True)
    print(model_fit.summary())
    wt_results = np.array(
        [comp for idx, comp in enumerate(all_components) if mouse_id[idx][:3] == "000"]
    )
    nlgf_results = np.array(
        [comp for idx, comp in enumerate(all_components) if mouse_id[idx][:3] != "000"]
    )
    # wt_results = gaussian_filter1d(wt_results, sigma=2, axis=1)
    # nlgf_results = gaussian_filter1d(nlgf_results, sigma=2, axis=1)

    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    plt.figure()
    shaded_line_plot(wt_results, x_axis=x_axis, color=palette["WT"], label="WT")
    shaded_line_plot(
        nlgf_results, x_axis=x_axis, color=palette["NLGF"], label="NLGF/S305N"
    )

    x_lim = (-2, 2)

    plt.legend()
    plt.xlabel("Time from ripple peak (s)")
    plt.ylabel("Reactivation strength (z-score)")
    plt.xlim(x_lim)
    sns.despine()
    plt.title("Mean component traces")
    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / "spontaneous_reactivation" / "mean_components.png", dpi=300
    )

    plt.figure()

    # boxplot averages across mouse
    sns.boxplot(
        data=df.groupby(["mouse", "genotype"]).mean().reset_index(),
        x="genotype",
        y="result",
        showfliers=False,
    )

    sns.stripplot(
        data=df.groupby(["mouse", "genotype"]).mean().reset_index(),
        x="genotype",
        y="result",
    )

    # sns.boxplot(data=df, x="genotype", y="result")

    plt.figure()

    labelled = False
    for comp in wt_results:
        plt.plot(
            x_axis,
            comp,
            color=colors[0],
            alpha=0.5,
            label="WT" if not labelled else None,
        )
        labelled = True

    labelled = False
    for comp in nlgf_results:
        plt.plot(
            x_axis,
            comp,
            color=colors[1],
            alpha=0.5,
            label="NLGF/S305N" if not labelled else None,
        )
        labelled = True

    plt.legend()

    plt.xlabel("Time from ripple peak (s)")
    plt.ylabel("Reactivation strength (z-score)")
    plt.xlim(-0.5, 0.5)
    sns.despine()
    plt.title("All component traces")
    plt.tight_layout()
    plt.savefig(
        FIGURE_PATH / "spontaneous_reactivation" / "all_components.png", dpi=300
    )

    1 / 0


def load_spindles(
    mouse: str,
    imec: str,
) -> Tuple[np.ndarray, np.ndarray]:

    spindle_path = HERE / "results" / "spindles"

    with open(spindle_path / f"{mouse}_imec_{imec[-1]}.json") as f:
        spindle_cache = SpindleCache.model_validate_json(f.read())

    return spindle_cache.spindles, spindle_cache.state


def load_ripples(
    mouse: str, imec: str, session_type: Literal["conditioning", "resting", "tones"]
) -> Tuple[np.ndarray, np.ndarray]:

    with open(RIPPLE_PATH / f"{mouse}_imec_{imec[-1]}_{session_type}.json") as f:
        ripple_cache = RipplesCache.model_validate_json(f.read())

    passing_checks = (
        np.array(ripple_cache.common_average_reference_check_less_restrictive)
        & np.array(ripple_cache.frequency_check)
        & np.array(ripple_cache.super_ripple_check_less_restrictive)
    )
    if mouse == "00053":
        passing_checks = passing_checks[: len(ripple_cache.candidate_events)]
        ripple_cache.state = ripple_cache.state[: len(ripple_cache.candidate_events)]

    ripples = np.array(ripple_cache.candidate_events)[passing_checks]
    state = np.array(ripple_cache.state)[passing_checks]
    return ripples, state


def align_reactivation_to_ripples(
    mouse: str,
    imec: str,
    lfp_sync: np.ndarray,
    spikes_sync: np.ndarray,
    pycontrol_resting_time_edges: Tuple[float, float],
    sessions: List[Session],
    resting_bin_edges: np.ndarray,
    reactivation_strength: np.ndarray,
    alignment_point: Literal["onset", "peak"],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    # reactivation_strength = np.mean(reactivation_strength, axis=0, keepdims=True)
    ripples, state = load_ripples(mouse, imec, session_type="resting")
    # ripples, state = load_spindles(mouse, imec)
    ripple_times_spikes = get_ripple_times_in_spikes(
        spikes_sync=spikes_sync,
        lfp_sync=lfp_sync,
        pycontrol_time_edges=pycontrol_resting_time_edges,
        sessions=sessions,
        ripples=ripples,
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

        closest_bin = np.argmin(
            np.abs(resting_bin_edges - r[0 if alignment_point == "onset" else 2])
        )
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

    if len(n_rem_trials) < 1:
        return np.array([]), np.array([])

    components = []
    raw_components = []
    for component in range(n_rem_trials.shape[1]):
        assembly_data = n_rem_trials[:, component, :]
        # compute global mean/std across ripples and time
        mean = assembly_data.mean()
        std = assembly_data.std(ddof=1)
        # zscore the whole (n_ripples, n_time) block
        zdata = (assembly_data - mean) / std
        # average across ripples -> one trace per assembly
        # components.append(np.abs(zdata.mean(axis=0)))
        components.append(zdata.mean(axis=0))
        raw_components.append(zdata)
    return components, raw_components


def get_ripple_times_in_spikes(
    spikes_sync: np.ndarray,
    lfp_sync: np.ndarray,
    pycontrol_time_edges: Tuple[float, float],
    sessions: List[Session],
    ripples: List[CandidateEvent] | np.ndarray,
) -> np.ndarray:

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
    first_sample = lfp_pycontrol_aligner.B_to_A(np.array([pycontrol_time_edges[0]]))[0]

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

    return ripple_times_spikes


def get_reactivation_strength(
    data_path: Path,
    kilosort_path: Path,
    region_boundaries: Tuple[int, int, int, int],
    pycontrol_conditioning_time_edges: Tuple[float, float],
    pycontrol_resting_time_edges: Tuple[float, float],
    ripple_times_spikes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    ca1_low, ca1_high, rsc_low, rsc_high = region_boundaries

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

    # Remove spikes that occur during ripples
    for ripple in ripple_times_spikes:
        idx = (spike_times >= ripple[0]) & (spike_times <= ripple[1])
        prev_length = spike_times.shape[0]
        spike_times = spike_times[~idx]
        spike_clusters = spike_clusters[~idx]
        closest_channel = closest_channel[~idx]
        print(
            "number of spikes removed during ripples:",
            prev_length - spike_times.shape[0],
        )

    bin_width = 0.02
    ssp_vectors, _ = build_cluster_matrix(
        spike_times, spike_clusters, start_conditioning, end_conditioning, bin_width
    )

    cluster_regions = []
    for cluster_id in np.unique(spike_clusters):
        closest_channel_cluster = closest_channel[spike_clusters == cluster_id][0]
        if closest_channel_cluster >= ca1_low and closest_channel_cluster <= ca1_high:
            cluster_regions.append("ca1")
        elif closest_channel_cluster >= rsc_low and closest_channel_cluster <= rsc_high:
            cluster_regions.append("rsc")
        else:
            raise ValueError("Cluster not in ca1 or rsc")

    cluster_regions = np.array(cluster_regions)

    # Sometimes get a cell that never spiked
    clusters_keep = np.sum(ssp_vectors, axis=1) > 0
    # Ripple silencing could create artifacts, so remove these
    timepoints_keep = np.sum(ssp_vectors, axis=0) > 0
    ssp_vectors = ssp_vectors[clusters_keep, :]
    ssp_vectors = ssp_vectors[:, timepoints_keep]
    # comps = compute_ICA_components(ssp_vectors)

    # Done in the same way as the lfp
    start_rest, end_rest = aligners[1].B_to_A(
        np.array([pycontrol_resting_time_edges[0], pycontrol_resting_time_edges[1]])
    )

    reactivation, resting_bin_edges = build_cluster_matrix(
        spike_times, spike_clusters, start_rest, end_rest, bin_width
    )
    return reactivation, resting_bin_edges

    reactivation = reactivation[clusters_keep, :]

    assert reactivation.shape[0] == ssp_vectors.shape[0]
    reactivation_strength = offline_reactivation(
        reactivation,
        comps,
        do_shuffle=False,
        cluster_regions=cluster_regions,
    )

    pcc_scores = compute_pcc_scores(reactivation, comps)

    return reactivation_strength, resting_bin_edges, pcc_scores


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


def fraction_positive_negative() -> None:
    with open("all_raw_components.pkl", "rb") as f:
        all_raw_components = pickle.load(f)
    with open("all_components.pkl", "rb") as f:
        all_components = np.array(pickle.load(f))

    resting_bin_edges = np.load(SERVER_CACHE_PATH / "00053_imec0_binedges.npy")
    mouse_id = np.load("all_mouse_ids.npy")

    assert len(mouse_id) == len(all_components)

    bin_size = 600
    trial_size = (30_000 * 8) // bin_size
    x_axis = np.arange(-trial_size, trial_size) * bin_size / 30_000
    ripple_start = np.where(x_axis == -0.1)[0][0]
    ripple_end = np.where(x_axis == 0.1)[0][0]
    baseline_start = np.where(x_axis == -2)[0][0]
    baseline_end = np.where(x_axis == -1)[0][0]

    positives = []
    negatives = []
    for component in all_raw_components:
        ripple_response = component[:, ripple_start:ripple_end]
        baseline_response = component[:, baseline_start:baseline_end]
        # assert ripple_response.shape == baseline_response.shape
        p_values = ttest_ind(ripple_response, baseline_response, axis=1).pvalue
        direction = np.sign(
            np.mean(ripple_response, axis=1) - np.mean(baseline_response, axis=1)
        )

        significant_positive = np.sum((p_values < 0.05) & (direction > 0))
        significant_negative = np.sum((p_values < 0.05) & (direction < 0))
        positives.append(significant_positive / component.shape[0])
        negatives.append(significant_negative / component.shape[0])
    positives = np.array(positives)
    negatives = np.array(negatives)

    df = pd.DataFrame(
        {
            "positive": positives,
            "negative": negatives,
            "mouse": mouse_id,
            "genotype": ["WT" if m[:3] == "000" else "NLGF/S305N" for m in mouse_id],
        }
    )
    colors = sns.color_palette(n_colors=2)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    sns.boxplot(
        data=df,
        x="genotype",
        y="positive",
        showfliers=False,
        color=colors[0],
        ax=axes[0],
        # palette=colors,
    )
    sns.stripplot(
        data=df,
        x="genotype",
        y="positive",
        dodge=True,
        jitter=True,
        size=3,
        color=colors[0],
        ax=axes[0],
    )

    sns.boxplot(
        data=df,
        x="genotype",
        y="negative",
        showfliers=False,
        color=colors[0],
        ax=axes[1],
    )

    sns.stripplot(
        data=df,
        x="genotype",
        y="negative",
        dodge=True,
        jitter=True,
        size=3,
        color=colors[0],
        ax=axes[1],
    )
    axes[0].set_title("Positive")
    axes[1].set_title("Negative")
    axes[0].set_ylabel("Fraction significant")
    sns.despine()
    plt.tight_layout()

    mixed_effect_positive = smf.mixedlm(
        "positive ~ genotype",
        df,
        groups=df["mouse"],
        use_sqrt=True,
    )
    mixed_effect_negative = smf.mixedlm(
        "negative ~ genotype",
        df,
        groups=df["mouse"],
        use_sqrt=True,
    )

    mixed_effect_positive_fit = mixed_effect_positive.fit(reml=True)
    mixed_effect_negative_fit = mixed_effect_negative.fit(reml=True)

    print("Positive components mixed effect model:")
    print(mixed_effect_positive_fit.summary())
    print("Negative components mixed effect model:")
    print(mixed_effect_negative_fit.summary())

    # Add the pvalue to the top of the plots
    axes[0].text(
        0.5,
        0.85,
        f"p = {mixed_effect_positive_fit.pvalues['genotype[T.WT]']:.2g}",
        ha="center",
        va="bottom",
        fontsize=14,
    )
    axes[1].text(
        0.5,
        0.85,
        f"p = {mixed_effect_negative_fit.pvalues['genotype[T.WT]']:.2g}",
        ha="center",
        fontsize=14,
        va="bottom",
    )
    plt.ylim(None, 0.95)

    # Draw a line under the pvalue joining the boxes

    axes[0].plot([0, 1], [0.83, 0.83], color="black", lw=1.5)
    axes[1].plot([0, 1], [0.83, 0.83], color="black", lw=1.5)

    # with a little vertical line at one end
    axes[0].plot([0, 0], [0.81, 0.83], color="black", lw=1.5)
    axes[0].plot([1, 1], [0.81, 0.83], color="black", lw=1.5)

    axes[1].plot([0, 0], [0.81, 0.83], color="black", lw=1.5)
    axes[1].plot([1, 1], [0.81, 0.83], color="black", lw=1.5)
    SERVER_PATH = Path("/Volumes/MarcBusche/James/figures")
    plt.savefig(
        SERVER_PATH / "spontaneous_reactivation" / "positive_negative_components.png"
    )


if __name__ == "__main__":

    fraction_positive_negative()
    # main()

    # with open("all_components.pkl", "rb") as f:
    #     all_components = pickle.load(f)

    # with open("all_raw_components.pkl", "rb") as f:
    #     all_raw_components = pickle.load(f)

    # resting_bin_edges = np.load(SERVER_CACHE_PATH / "00053_imec0_binedges.npy")
    # mouse_ids_components = np.load("all_mouse_ids.npy")
    # plot_component_traces(all_components, mouse_ids_components, resting_bin_edges)
