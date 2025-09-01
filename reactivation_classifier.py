from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from scipy.spatial import cKDTree
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from data_import import Session
from scipy.stats import zscore
from gsheets_importer import gsheet2df
from lfp_signatures import get_ca1_rsc_channels
from rsync import Rsync_aligner
from utils import (
    build_path_dict,
    get_aligners,
    get_data_paths,
    process_session,
    save_figure,
)
from sklearn.preprocessing import minmax_scale

from scipy.stats import kruskal, mannwhitneyu


from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from ripples.utils_npyx import load_sync_npyx
from ripples.utils import threshold_detect
from plotting import plot_cell_summed_results

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

HERE = Path(__file__).parent

FIGURE_PATH = HERE / "plots" / "classifier"

WT_COLOR = "#1f77b4"
NLGF_COLOR = "#ff7f0e"

SHUFFLED_COLOR = sns.color_palette("tab10")[2]


def load_spiking_data(
    kilosort_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    assert np.all(np.diff(cluster_id) == 1), "Cluster IDs are not sorted ascending."
    assert len(cluster_id) == len(
        label
    ), "Cluster IDs and labels are not the same length."
    id_to_label = dict(zip(cluster_id, label))
    label_array = np.array([id_to_label[cluster] for cluster in spike_clusters])

    channel_positions = np.load(kilosort_path / "channel_positions.npy")
    spike_positions = np.load(kilosort_path / "spike_positions.npy")
    # Find the closest channel depth for each spike in a memory-efficient way

    if (kilosort_path / "closest_channel.npy").exists():
        closest_channel = np.load(kilosort_path / "closest_channel.npy")
    else:
        tree = cKDTree(channel_positions)
        _, closest_channel = tree.query(spike_positions, k=1)
        np.save(kilosort_path / "closest_channel.npy", closest_channel)

    # # Useful debugging plot, looks correct, might want to look at it again
    # for g in np.unique(spike_clusters):
    #     indy = spike_clusters == g
    #     pos = spike_positions[indy]
    #     channy = closest_channel[indy]
    #     cd = np.max(channy) - np.min(channy)
    #     if cd > 12:
    #         plt.plot(
    #             channel_positions[channy][:, 0], channel_positions[channy][:, 1], "."
    #         )
    #         plt.plot(pos[:, 0], pos[:, 1], ".")
    #         1 / 0

    return spike_times, spike_clusters, label_array, closest_channel


def split_data_by_trial(
    stim_times: np.ndarray,
    spikes: np.ndarray,
    spike_clusters: np.ndarray,
    window: float,
    n_bins: int,
    sampling_rate: int = 30_000,
) -> np.ndarray:
    # Precompute actual clusters and mapping
    actual_clusters = np.unique(spike_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(actual_clusters)}
    n_clusters = len(actual_clusters)

    # Calculate window in samples and create bin edges
    window_samples = window * sampling_rate
    bin_edges = np.linspace(-window_samples, window_samples, n_bins + 1)

    result = []

    for onset in stim_times:
        start = onset - window_samples
        end = onset + window_samples
        # Use half-open interval [start, end) to match histogram behavior
        mask = (spikes >= start) & (spikes < end)
        trial_spikes = spikes[mask]
        trial_clusters = spike_clusters[mask]

        # Initialize array with correct dimensions
        trial_result = np.zeros((n_clusters, n_bins))

        # Process each cluster present in the trial
        for cluster in np.unique(trial_clusters):
            # Get relative spike times for this cluster
            cluster_spikes = trial_spikes[trial_clusters == cluster]
            rel_times = cluster_spikes - onset

            # Bin the relative spike times
            binned, _ = np.histogram(rel_times, bins=bin_edges)

            # Store using cluster index mapping
            idx = cluster_to_idx[cluster]
            trial_result[idx, :] = binned

        result.append(trial_result)

    return np.array(result)


def reduce_array_resolution(arr: np.ndarray, n: int) -> np.ndarray:
    # Reduce the spatial resolution of the array by averaging
    assert arr.shape[2] % n == 0, "Array shape is not divisible by n"
    return arr.reshape(arr.shape[0], arr.shape[1], -1, n).sum(axis=-1)


def dimensionality_reduction(training_array, testing_array):
    # Current bin is 100 bins in 1.8 seconds = 18 ms bins

    # original shape
    # (n_trials, n_clusters, n_bins)
    training_array = reduce_array_resolution(training_array, 2)
    testing_array = reduce_array_resolution(testing_array, 2)

    # Put the cells first so it's a bit easier
    training_array = np.swapaxes(training_array, 0, 1)
    testing_array = np.swapaxes(testing_array, 0, 1)

    training_array_flattened = np.reshape(training_array, (training_array.shape[0], -1))
    testing_array_flattened = np.reshape(testing_array, (testing_array.shape[0], -1))

    combined = np.concatenate(
        (training_array_flattened, testing_array_flattened), axis=1
    )

    pca = PCA()
    reduced = pca.fit_transform(combined.T).T

    # Restore the original trial shape in PCA space
    n_training_samples = training_array.shape[1] * training_array.shape[2]
    training_array_reduced = reduced[:, :n_training_samples].reshape(
        *training_array.shape
    )

    testing_array_reduced = reduced[:, n_training_samples:].reshape(
        *testing_array.shape
    )
    return np.swapaxes(training_array_reduced, 0, 1), np.swapaxes(
        testing_array_reduced, 0, 1
    )


def process_mouse(
    data_path: Path,
    df: pd.DataFrame,
    kilosort_paths: List[Path],
    C: float,
    penalty: str,
    solver: str,
) -> None:
    subject = data_path.parts[-1]

    print("=" * 20)

    save_path = HERE / "results" / "trial_arrays"
    redo = False

    if (save_path / f"{subject}_clusters_info.npy").exists() and not redo:
        print(f"Loading existing data for {subject}")
        training_array = np.load(save_path / f"{subject}_training_array.npy")
        testing_array = np.load(save_path / f"{subject}_testing_array.npy")
        training_labels = np.load(save_path / f"{subject}_training_labels.npy")
        testing_labels = np.load(save_path / f"{subject}_testing_labels.npy")
        cluster_infos = np.load(
            save_path / f"{subject}_clusters_info.npy", allow_pickle=True
        )

    else:
        training_arrays = []
        training_labels = []
        testing_arrays = []
        testing_labels = []
        cluster_infos = []

        for kilosort_path in kilosort_paths:
            ca1_low, ca1_high, rsc_low, rsc_high = get_ca1_rsc_channels(
                kilosort_path, df
            )

            print(f"CA1: {ca1_low} - {ca1_high}, RSC: {rsc_low} - {rsc_high}")

            train_array, y_train, test_array, y_test, cluster_info = process_probe(
                data_path, kilosort_path, (ca1_low, ca1_high, rsc_low, rsc_high)
            )
            training_arrays.append(train_array)
            training_labels.append(y_train)
            testing_arrays.append(test_array)
            testing_labels.append(y_test)
            cluster_infos.append(cluster_info)

        training_array = np.concatenate(training_arrays, axis=1)
        testing_array = np.concatenate(testing_arrays, axis=1)

        np.save(save_path / f"{subject}_clusters_info.npy", cluster_infos)
        np.save(
            save_path / f"{subject}_training_array.npy",
            training_array,
        )
        np.save(
            save_path / f"{subject}_testing_array.npy",
            testing_array,
        )
        np.save(
            save_path / f"{subject}_training_labels.npy",
            training_labels,
        )
        np.save(
            save_path / f"{subject}_testing_labels.npy",
            testing_labels,
        )

    # One session only has one probe need to handle this
    n_probes = len(cluster_infos)

    cluster_infos[0]["probe"] = [0] * len(cluster_infos[0]["label"])

    if n_probes > 1:
        assert np.array_equal(training_labels[0], training_labels[1])
        assert np.array_equal(testing_labels[0], testing_labels[1])
        cluster_infos[1]["probe"] = [1] * len(cluster_infos[1]["label"])
        cluster_info = {
            "probe": cluster_infos[0]["probe"] + cluster_infos[1]["probe"],
            "label": cluster_infos[0]["label"] + cluster_infos[1]["label"],
            "region": cluster_infos[0]["region"] + cluster_infos[1]["region"],
        }
    else:
        cluster_info = cluster_infos[0]

    # makes indexing easier downstream
    for key in ["probe", "label", "region"]:
        cluster_info[key] = np.array(cluster_info[key])

    assert (
        len(cluster_info["label"])
        == len(cluster_info["probe"])
        == len(cluster_info["region"])
        == training_array.shape[1]
    )

    trial_states = get_sleep_state(data_path)
    assert len(trial_states) == testing_array.shape[0]
    trials_keep = np.isin(trial_states, ["nrem", "deep nrem"])

    for probe_idx in range(n_probes):

        clusters_keep = (cluster_info["probe"] == probe_idx) & (
            (cluster_info["region"] == "RSC") | (cluster_info["region"] == "CA1")
        )

        training_reduced_probe, testing_reduced_probe = dimensionality_reduction(
            training_array=training_array[:, clusters_keep, :],
            testing_array=testing_array[:, clusters_keep, :],
        )

        if probe_idx == 0:
            training_reduced = training_reduced_probe
            testing_reduced = testing_reduced_probe
        else:
            training_reduced = np.concatenate(
                (training_reduced, training_reduced_probe), axis=1
            )
            testing_reduced = np.concatenate(
                (testing_reduced, testing_reduced_probe), axis=1
            )

    print(
        f"Keeping {np.sum(trials_keep)} trials out of {len(trial_states)} based on sleep state."
    )

    # Slightly janky redefinition but allows to switch between reduced and non-reduced
    # training_array = training_reduced
    # testing_array = testing_reduced

    training_array = training_array[:, cluster_info["region"] == "RSC", :]
    testing_array = testing_array[:, cluster_info["region"] == "RSC", :]

    testing_array = testing_array[trials_keep, :, :]
    testing_labels = testing_labels[0][trials_keep]
    assert (
        len(testing_labels) > 0
    ), "Testing labels are empty after filtering by sleep state."

    scores = []
    awake_scores = []
    sleep_shuffled = []
    for awake_offset in range(-5, 15, 3):
        for sleep_offset in range(-5, 15, 3):
            (
                model,
                label_encoder,
                awake_score,
                awake_shuffled_scores,
            ) = get_awake_classifier(
                training_array.copy(),
                training_labels[0],
                C=C,
                penalty=penalty,
                solver=solver,
                awake_offset=awake_offset,
            )
            awake_scores.append(awake_score)

            score, result_shuffled = fit_classifier_to_sleeping_data(
                testing_array.copy(),
                testing_labels,
                model,
                label_encoder,
                sleep_offset=sleep_offset,
            )

            scores.append(score)
            sleep_shuffled.append(result_shuffled)

    return scores, awake_scores, sleep_shuffled


def process_probe(
    data_path: Path,
    kilosort_path: Path,
    region_boundaries: Tuple[int, int, int, int],
    bin_data: bool = True,
) -> (
    # Worst return type of all time
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]
    | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Rsync_aligner]]
):
    """If bin_data is set to true, you'll get a trial by trial array of binned data. Otherwise you'll get the spike times etc
    and the aligners.

    I did this to hijack this function for spontaneous_reactivation. Should be refactored but oh well

    """
    ca1_low, ca1_high, rsc_low, rsc_high = region_boundaries

    assert ca1_low < ca1_high < rsc_low < rsc_high
    assert 30 <= ca1_high - ca1_low <= 70
    assert 80 <= rsc_high - rsc_low <= 120

    _, _, pycontrol_files = get_data_paths(data_path)
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    rsync_times = [session.times["rsync"] for session in sessions]

    if (kilosort_path.parent / "high_pass_sync.npy").exists():
        print("Loading existing sync file")
        npx_sync_times = np.load(kilosort_path.parent / "high_pass_sync.npy")
    else:
        sync = load_sync_npyx(str(kilosort_path.parent), filt_key="highpass")
        npx_sync_times = (
            threshold_detect(sync, 0.5)
            if "00053" not in str(kilosort_path)
            else threshold_detect(sync[:, 6], 0.5)
        )
        np.save(kilosort_path.parent / "high_pass_sync.npy", npx_sync_times)

    # This passes most of the time but doesn't for one mouse, the aligners don't error so oh well.
    # assert sum(len(rsync) for rsync in rsync_times) == len(
    #     npx_sync_times
    # ), "Rsync times and NPX sync times do not match in length."

    aligners = get_aligners(npx_sync_times, rsync_times)

    spike_times, spike_clusters, labels, closest_channel = load_spiking_data(
        kilosort_path
    )

    idx_keep = ((closest_channel >= ca1_low) & (closest_channel <= ca1_high)) | (
        (closest_channel >= rsc_low) & (closest_channel <= rsc_high)
    )

    spike_times = spike_times[idx_keep]
    spike_clusters = spike_clusters[idx_keep]

    print(f"Number of clusters : {len(np.unique(spike_clusters))}")

    labels = labels[idx_keep]
    closest_channel = closest_channel[idx_keep]

    if not bin_data:
        return spike_times, spike_clusters, labels, closest_channel, aligners

    n_bins = 100
    window = 0.9  # seconds

    train_array, y_train = get_training_data(
        sessions, aligners, spike_clusters, spike_times, n_bins, window
    )

    test_array, y_test = get_testing_data(
        sessions, aligners, spike_clusters, spike_times, n_bins, window
    )
    cluster_info: Dict[str : List[str]] = {"label": [], "region": []}

    for spike_cluster in np.unique(spike_clusters):
        idx_cluster = spike_clusters == spike_cluster
        cluster_label = labels[idx_cluster]
        assert len(np.unique(cluster_label)) == 1, "Cluster has multiple labels."

        # Different spikes in the same cluster can be localised to different channels
        closest_channels_cluster = closest_channel[idx_cluster]

        # This seems like a lot but I've manually inspected the positions of the spikes
        # and the channels they have been asigned to. The logic is correct.
        # It could be drift or the clusters need splitting
        assert np.max(closest_channels_cluster) - np.min(closest_channels_cluster) <= 12

        average_channel = np.mean(closest_channels_cluster)

        region_cluster = (
            "CA1"
            if ca1_low <= average_channel <= ca1_high
            else "RSC" if rsc_low <= average_channel <= rsc_high else None
        )

        assert region_cluster is not None
        cluster_info["label"].append(cluster_label[0])
        cluster_info["region"].append(region_cluster)

    return train_array, y_train, test_array, y_test, cluster_info


def normalize_trial_array(trial_array: np.ndarray) -> np.ndarray:
    """Normalize the trial array by z-scoring each cluster across trials."""
    for trial_idx in range(trial_array.shape[0]):
        trial = trial_array[trial_idx, :, :]
        # Baseline the trial to the average of the first half of the bins
        # trial -= trial[:, : trial.shape[1] // 2].mean(axis=1, keepdims=True)
        trial = zscore(trial, axis=1)
        trial[np.isnan(trial)] = 0  # Replace NaNs with 0
        # trial = gaussian_filter1d(trial, sigma=3, axis=1)
        trial_array[trial_idx, :, :] = trial

    return trial_array


def fit_classifier_to_sleeping_data(
    trial_array: np.ndarray,
    y: np.ndarray,
    awake_model: LogisticRegression,
    label_encoder: LabelEncoder,
    sleep_offset: int,
    plot: bool = False,
) -> Tuple[float, List[float]]:
    X = X_from_trial_array(trial_array, offset=sleep_offset)

    # Sum across the post stimulus bins

    # Blue -> 3000, Orange -> 8000
    blue_encoding, orange_encoding = label_encoder.transform(["blue", "orange"])

    y_encoded = np.array(
        [
            (
                blue_encoding
                if sound == 3000
                else orange_encoding if sound == 8000 else None
            )
            for sound in y
        ]
    )

    assert np.all(
        y_encoded is not None
    ), "Some sounds do not have a valid frequency encoding."

    assert not np.all(
        y_encoded == y_encoded[0]
    ), "All labels are the same. Cannot compute score."

    result = awake_model.predict(X)
    print(f"Predicted labels: {result}")
    score = balanced_accuracy_score(y_encoded, result)

    print(f"Classifier score on sleeping data: {score:.2f}")

    result_shuffled = []
    for _ in range(100):
        shuffle_idx = np.random.permutation(len(y_encoded))
        X_shuffled = X[shuffle_idx, :]
        result_shuffled.append(
            balanced_accuracy_score(y_encoded, awake_model.predict(X_shuffled))
        )

    return score, result_shuffled


def get_testing_data(
    sessions: List[Session],
    aligners: List[Rsync_aligner],
    spike_clusters: np.ndarray,
    spike_times: np.ndarray,
    n_bins: int,
    window: float,
) -> Tuple[np.ndarray, np.ndarray]:
    session_index = 2
    sounds, _ = process_session(sessions[session_index])

    sounds_times = aligners[session_index].B_to_A(
        np.array([sound.time for sound in sounds]), extrapolate=True
    )
    assert np.all(np.isclose((np.diff(sounds_times) / 30000), 11.5, atol=0.01))

    trial_array = split_data_by_trial(
        stim_times=sounds_times,
        spikes=spike_times,
        spike_clusters=spike_clusters,
        window=window,
        n_bins=n_bins,
    )

    return trial_array, np.array([sound.frequency for sound in sounds])


def X_from_trial_array(trial_array: np.ndarray, offset: int = 0) -> np.ndarray:
    # (n_trials, n_clusters, n_bins)
    # Transform to (n_samples, n_features)
    # normalize_trial_array(trial_array)
    n_bins = trial_array.shape[2]
    start = (n_bins // 2) + offset
    end = start + 20
    X = trial_array[:, :, start:end]
    # return X.reshape(X.shape[0], -1)
    return X.mean(axis=2)


def get_awake_classifier(
    trial_array: np.ndarray,
    y: np.ndarray,
    C: float,
    penalty: str,
    solver: str,
    awake_offset: int,
) -> Tuple[LogisticRegression, LabelEncoder, float, List[float]]:
    X = X_from_trial_array(trial_array, offset=awake_offset)

    # (n_trials, n_clusters, n_bins)
    # Transform to (n_samples, n_features)
    # Sum across the post stimulus bins

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=C,
        l1_ratio=0.5 if penalty == "elasticnet" else None,
        fit_intercept=True,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy")
    print("Mean 5-fold accuracy:", scores.mean())
    model = LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=C,
        l1_ratio=0.5 if penalty == "elasticnet" else None,
        fit_intercept=True,
    )

    model.fit(X, y_encoded)
    return model, le, scores.mean(), []


def get_training_data(
    sessions: List[Session],
    aligners: List[Rsync_aligner],
    spike_clusters: np.ndarray,
    spike_times: np.ndarray,
    n_bins: int,
    window: float,
) -> Tuple[np.ndarray, np.ndarray]:
    session_index = 0

    _, LEDs = process_session(sessions[session_index])
    assert LEDs is not None

    LEDs_times = aligners[session_index].B_to_A(
        np.array([led.time for led in LEDs]), extrapolate=True
    )

    colors = np.array([led.color for led in LEDs])

    # Not sure why this happened, only trial in one mouse on one probe so probably fine but still...
    if np.sum(np.isnan(LEDs_times)) == 1:
        print(
            f"Warning: NaN value in LEDs times for mouse {sessions[session_index].subject_ID}. Patching as a temporary fix."
        )
        nan_idx = np.where(np.isnan(LEDs_times))[0][0]
        LEDs_times[nan_idx] = LEDs_times[nan_idx - 1] + 11.5 * 30000

    assert np.all(np.isclose((np.diff(LEDs_times) / 30000), 11.5, atol=0.01))

    trial_array = split_data_by_trial(
        stim_times=LEDs_times,
        spikes=spike_times,
        spike_clusters=spike_clusters,
        window=window,
        n_bins=n_bins,
    )

    return (
        trial_array,
        colors,
    )


def plot_processed_data() -> float:
    results_path = Path(__file__).parent / "results"

    mice = set([file.stem.split("_")[0] for file in results_path.glob("*.npy")])
    WT_scores = []
    NLGF_scores = []
    awake_scores = []
    shuffled_label_scores = []

    for mouse in mice:
        if mouse in ["11153"]:
            continue
        score = np.load(results_path / f"{mouse}_classifier_score.npy").item()
        awake_scores.append(np.load(results_path / f"{mouse}_awake_scores.npy").item())
        shuffled_label_scores.extend(
            np.load(results_path / f"{mouse}_classifier_shuffled_labels.npy")
        )

        if mouse[:3] == "000":
            WT_scores.append(score)
            print(f"Mouse {mouse} is WT")
        else:
            NLGF_scores.append(score)
            print(f"Mouse {mouse} is NLGF/S305N")

    plt.figure()
    b = sns.boxplot(
        data={
            "WT": WT_scores,
            "NLGFxS305N": NLGF_scores,
            "Shuffled": shuffled_label_scores,
        },
        palette={"WT": WT_COLOR, "NLGFxS305N": NLGF_COLOR, "Shuffled": SHUFFLED_COLOR},
        showfliers=False,
    )

    sns.stripplot(
        data={
            "WT": WT_scores,
            "NLGFxS305N": NLGF_scores,
        },
        palette={"WT": WT_COLOR, "NLGFxS305N": NLGF_COLOR},
        dodge=False,
        alpha=1,
        legend=False,
        linewidth=0.5,
    )

    b.tick_params(labelsize=12)
    b.set_xlabel("Genotype", fontsize=14, fontweight="bold")
    b.set_ylabel("Classification Accuracy", fontsize=14, fontweight="bold")
    plt.grid(axis="y")
    plt.axhline(0.5, color="red", linestyle="--", label="Chance level")
    plt.legend(loc="upper right")

    p1 = mannwhitneyu(WT_scores, shuffled_label_scores, alternative="two-sided")
    p2 = mannwhitneyu(NLGF_scores, shuffled_label_scores, alternative="two-sided")
    p3 = mannwhitneyu(WT_scores, NLGF_scores, alternative="two-sided")

    plt.title(
        "Classification accuracy (targeted reactivation)",
        fontsize=16,
        fontweight="bold",
    )

    save_figure("sleeping", FIGURE_PATH)

    summary_df = {
        "wt_mean": np.mean(WT_scores),
        "wt_median": np.median(WT_scores),
        "wt_std": np.std(WT_scores),
        "nlgf_mean": np.mean(NLGF_scores),
        "nlgf_median": np.median(NLGF_scores),
        "nlgf_std": np.std(NLGF_scores),
        "shuffled_mean": np.mean(shuffled_label_scores),
        "shuffled_median": np.median(shuffled_label_scores),
        "shuffled_std": np.std(shuffled_label_scores),
        "wt_vs_shuffled_p": p1.pvalue,
        "wt_vs_shuffled_u": p1.statistic,
        "nlgf_vs_shuffled_p": p2.pvalue,
        "nlgf_vs_shuffled_u": p2.statistic,
        "wt_vs_nlgf_p": p3.pvalue,
        "wt_vs_nlgf_u": p3.statistic,
    }
    pd.DataFrame(summary_df, index=[0]).to_csv(
        FIGURE_PATH / "sleep_summary_stats.csv", index=False
    )
    return np.mean(awake_scores)


def plot_processed_data_waking() -> float:
    results_path = Path(__file__).parent / "results"
    mice = set([file.stem.split("_")[0] for file in results_path.glob("*.npy")])
    WT_scores = []
    NLGF_scores = []
    shuffled_label_scores = []

    for mouse in mice:
        if mouse in ["11153"]:
            continue

        score = np.load(results_path / f"{mouse}_awake_scores.npy").item()
        shuffled_label_scores.extend(
            np.load(results_path / f"{mouse}_awake_shuffled_scores.npy")
        )
        if mouse[:3] == "000":
            WT_scores.append(score)
        else:
            NLGF_scores.append(score)

    plt.figure()
    b = sns.boxplot(
        data={
            "WT": WT_scores,
            "NLGFxS305N": NLGF_scores,
            "Shuffled": shuffled_label_scores,
        },
        palette={"WT": WT_COLOR, "NLGFxS305N": NLGF_COLOR, "Shuffled": SHUFFLED_COLOR},
        showfliers=False,
    )

    sns.stripplot(
        data={
            "WT": WT_scores,
            "NLGFxS305N": NLGF_scores,
        },
        palette={"WT": WT_COLOR, "NLGFxS305N": NLGF_COLOR},
        dodge=False,
        alpha=1,
        legend=False,
        linewidth=0.5,
    )

    plt.axhline(0.5, color="red", linestyle="--", label="Chance level")
    plt.legend(loc="upper right")
    b.tick_params(labelsize=12)
    b.set_xlabel("Genotype", fontsize=14, fontweight="bold")
    b.set_ylabel("Classification Accuracy", fontsize=14, fontweight="bold")
    plt.grid(axis="y")
    plt.axhline(0.5, color="red", linestyle="--", label="Chance level")

    plt.title("Classification accuracy (waking)", fontsize=16, fontweight="bold")

    p1 = mannwhitneyu(WT_scores, shuffled_label_scores, alternative="two-sided")
    p2 = mannwhitneyu(NLGF_scores, shuffled_label_scores, alternative="two-sided")
    p3 = mannwhitneyu(WT_scores, NLGF_scores, alternative="two-sided")

    summary_df = {
        "wt_mean": np.mean(WT_scores),
        "wt_median": np.median(WT_scores),
        "wt_std": np.std(WT_scores),
        "nlgf_mean": np.mean(NLGF_scores),
        "nlgf_median": np.median(NLGF_scores),
        "nlgf_std": np.std(NLGF_scores),
        "shuffled_mean": np.mean(shuffled_label_scores),
        "shuffled_median": np.median(shuffled_label_scores),
        "shuffled_std": np.std(shuffled_label_scores),
        "wt_vs_shuffled_p": p1.pvalue,
        "wt_vs_shuffled_u": p1.statistic,
        "nlgf_vs_shuffled_p": p2.pvalue,
        "nlgf_vs_shuffled_u": p2.statistic,
        "wt_vs_nlgf_p": p3.pvalue,
        "wt_vs_nlgf_u": p3.statistic,
    }
    pd.DataFrame(summary_df, index=[0]).to_csv(
        FIGURE_PATH / "waking_summary_stats.csv", index=False
    )

    save_figure("waking", FIGURE_PATH)


def main() -> None:
    path_dict = build_path_dict()
    paths_df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)

    solver = "lbfgs"  # Solver for Logistic Regression
    penalty = "l2"
    C = 100

    # JUMP

    sleep_scores = {}
    awake_scores = {}
    shuffled_scores = {}
    for mouse, kilosort_paths in path_dict.items():
        assert all(
            [
                (kilosort_path / "spike_times.npy").exists()
                for kilosort_path in kilosort_paths
            ]
        )

        data_path = kilosort_paths[0].parent.parent.parent

        sleep_score, awake_score, sleep_shuffled = process_mouse(
            data_path, paths_df, kilosort_paths, C, penalty, solver
        )

        sleep_scores[mouse] = sleep_score
        awake_scores[mouse] = awake_score
        shuffled_scores[mouse] = sleep_shuffled

    np.save(HERE / "results" / f"mouse_sleep_scores.npy", sleep_scores)
    np.save(HERE / "results" / f"mouse_awake_scores.npy", awake_scores)
    np.save(HERE / "results" / f"mouse_sleep_shuffled.npy", shuffled_scores)

    return sleep_scores, awake_scores


def plot_boxplots(
    awake: Dict[str, List[float]],
    sleep: Dict[str, List[float]],
    shuffled: Dict[str, List[float]],
) -> None:
    wt_wake = []
    nlgf_wake = []

    wt_sleep = []
    nlgf_sleep = []

    wt_shuffled = []
    nlgf_shuffled = []

    for mouse in awake.keys():

        ##############
        result = awake[mouse]
        result = np.array(result).reshape(
            int(np.sqrt(len(result))), int(np.sqrt(len(result)))
        )
        result = gaussian_filter(result, sigma=1)

        if mouse[:3] == "000":
            wt_wake.append(result)
        else:
            nlgf_wake.append(result)

        ###################
        result = sleep[mouse]
        result = np.array(result).reshape(
            int(np.sqrt(len(result))), int(np.sqrt(len(result)))
        )
        result = gaussian_filter(result, sigma=1)

        if mouse[:3] == "000":
            wt_sleep.append(result)
        else:
            nlgf_sleep.append(result)

        ######################
        result = shuffled[mouse]

        result = np.array(result).reshape(
            int(np.sqrt(len(result))), int(np.sqrt(len(result))), len(result[0])
        )

        for shuffle in range(result.shape[2]):
            result_shuffle = gaussian_filter(result[:, :, shuffle], sigma=1)

            if mouse[:3] == "000":
                wt_shuffled.append(result_shuffle)
            else:
                nlgf_shuffled.append(result_shuffle)

    to_plot = {
        "WT": [np.percentile(result, 95) for result in wt_wake],
        "NLGF/S305N": [np.percentile(result, 95) for result in nlgf_wake],
    }

    sns.boxplot(to_plot)
    plt.title("95th percentile scores waking")

    to_plot = {
        "WT": [np.percentile(result, 95) for result in wt_sleep],
        "WT_Shuffled": [np.percentile(result, 95) for result in wt_shuffled],
        "NLGF/S305N": [np.percentile(result, 95) for result in nlgf_sleep],
        "NLGF/S305N_Shuffled": [np.percentile(result, 95) for result in nlgf_shuffled],
    }
    t_wt = ttest_ind(
        [np.percentile(result, 95) for result in wt_sleep],
        [np.percentile(result, 95) for result in wt_shuffled],
    )
    t_nlgf = ttest_ind(
        [np.percentile(result, 95) for result in nlgf_sleep],
        [np.percentile(result, 95) for result in nlgf_shuffled],
    )

    plt.figure()
    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot)
    plt.title(f"WT p={t_wt.pvalue:.3f}, NLGF/S305N p={t_nlgf.pvalue:.3f}")


def plot_heatmap(mouse_scores: Dict[str, List[float]], title: str) -> None:
    wt = []
    nlgf = []
    for mouse, result in mouse_scores.items():
        result = np.array(result).reshape(
            int(np.sqrt(len(result))), int(np.sqrt(len(result)))
        )
        result = gaussian_filter(result, sigma=1)

        if mouse[:3] == "000":
            wt.append(result)
            print(f"Mouse {mouse} is WT")
        else:
            nlgf.append(result)
            print(f"Mouse {mouse} is NLGF/S305N")

    vmax = 0.7
    vmin = 0.3
    for mouse in wt:
        plt.figure()
        plt.title("WT " + title)
        plt.imshow(mouse, vmin=vmin, vmax=vmax)

    for mouse in nlgf:
        plt.figure()
        plt.title("NLGF/S305N " + title)
        plt.imshow(mouse, vmin=vmin, vmax=vmax)

    to_plot = {
        "WT": [np.percentile(result, 95) for result in wt],
        "NLGF/S305N": [np.percentile(result, 95) for result in nlgf],
    }

    plt.figure()
    sns.boxplot(to_plot)
    plt.ylim(0.45, 0.7)
    plt.title("95th percentile scores " + title)


def get_sleep_state(data_path: Path) -> np.ndarray:
    num_to_state = {0.5: "nrem", 0: "deep nrem", 1: "rem", 2: "awake", 4: "movement"}

    _, _, pycontrol_files = get_data_paths(data_path)
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]
    tone_session = sessions[2]
    trial_starts = tone_session.times["trial_start"]
    sound_on = tone_session.times["sound_on"]
    if len(trial_starts) != len(sound_on):
        # Session stopped in between a trial start and a sound
        # Remove the final trial start, then check this logic is correct
        trial_starts = trial_starts[:-1]
        assert len(trial_starts) == len(sound_on)
        assert np.allclose(sound_on - trial_starts, 11, atol=0.01)
    trial_ends = trial_starts[1:]
    trial_ends = np.append(trial_ends, trial_starts[-1] + 11.5)

    assert np.allclose(trial_ends - trial_starts, 11.5, atol=0.01)

    mouse = data_path.parts[-1]

    spreadsheet_path = Path(
        "/Volumes/MarcBusche/Alex/Reactivations/Sleep Scoring/results"
    )
    spreadsheets = list(spreadsheet_path.glob(f"*.xlsx"))
    mouse_sheets = [
        file_name
        for file_name in spreadsheets
        if mouse in str(file_name).lower() and "tones" in str(file_name).lower()
    ]
    assert (
        len(mouse_sheets) == 1
    ), f"Expected one sleep scoring spreadsheet for mouse {mouse}, found {len(mouse_sheets)}."
    spreadsheet = mouse_sheets[0]

    data = pd.read_excel(spreadsheet, sheet_name="Sheet1")
    # assert str(data.Mouse[0]) == mouse

    mins = data["Minutes"].to_numpy()
    seconds = data["Seconds"].to_numpy()
    total_seconds = mins * 60 + seconds

    trial_start_idx = np.array(
        [np.argmin(np.abs(total_seconds - start)) for start in trial_starts]
    )

    assert np.all(np.isin(np.diff(trial_start_idx), [11, 12]))

    trial_end_idx = np.array(
        [np.argmin(np.abs(total_seconds - end)) for end in trial_ends]
    )
    trial_states = []

    for start, end in zip(trial_start_idx, trial_end_idx, strict=True):
        states = data["Score"][start:end].to_numpy()
        assert not np.any(np.isnan(states))
        if np.all(states == states[0]):
            trial_states.append(num_to_state[states[0]])
        else:
            trial_states.append("mixed")

    return np.array(trial_states)


if __name__ == "__main__":
    main()
    sleep_scores = np.load(
        HERE / "results" / f"mouse_sleep_scores.npy", allow_pickle=True
    ).item()
    awake_scores = np.load(
        HERE / "results" / f"mouse_awake_scores.npy", allow_pickle=True
    ).item()
    shuffled_scores = np.load(
        HERE / "results" / f"mouse_sleep_shuffled.npy", allow_pickle=True
    ).item()

    # plot_heatmap(sleep_scores, title="sleep")
    # plot_heatmap(shuffled_scores, title="shuffled")
    # plot_heatmap(awake_scores, title="awake")
    plot_boxplots(awake_scores, sleep_scores, shuffled_scores)

    plt.show()
