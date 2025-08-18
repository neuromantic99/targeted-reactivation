from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from scipy.spatial import cKDTree
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from data_import import Session
from scipy.stats import zscore
from gsheets_importer import gsheet2df
from lfp_signatures import get_ca1_rsc_channels
from rsync import Rsync_aligner
from utils import get_aligners, get_data_paths, process_session, save_figure
from sklearn.preprocessing import minmax_scale

from scipy.stats import kruskal, mannwhitneyu


from sklearn.metrics import balanced_accuracy_score
from scipy.ndimage import gaussian_filter1d

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


def split_data_by_trial_old(
    stim_times: np.ndarray,
    spikes: np.ndarray,
    spike_clusters: np.ndarray,
    window: float,
    n_bins: int,
    sampling_rate: int = 30_000,
) -> np.ndarray:
    """Split the spike data into trials based on stimulus times.

    Returns: np.ndarray (n_trials, n_clusters, n_bins)
    """

    result = []
    bin_edges = np.linspace(-window * sampling_rate, window * sampling_rate, n_bins)

    for onset in stim_times:
        start = onset - window * sampling_rate
        end = onset + window * sampling_rate
        idx_trial = (spikes >= start) & (spikes <= end)
        trial_spikes = spikes[idx_trial]
        trial_clusters = spike_clusters[idx_trial]

        trial_result = np.zeros((max(spike_clusters) + 1, n_bins - 1))

        for cluster in np.unique(trial_clusters):
            cluster_spikes = trial_spikes[trial_clusters == cluster]
            binned = np.histogram(cluster_spikes - onset, bins=bin_edges)[0]
            trial_result[cluster, :] = binned

        result.append(trial_result)

    result = np.array(result)

    # The result array spans from 0 to max(spike_clusters) but lots of these might not be actual clusters
    # if spike_clusters has been filtered. So return the actual clusters.
    actual_clusters = np.unique(spike_clusters)

    return result[:, actual_clusters, :]


def process_mouse(
    data_path: Path,
    df: pd.DataFrame,
    kilosort_paths: List[Path],
    C: float,
    penalty: str,
    solver: str,
    offset: int = 0,
) -> None:

    subject = data_path.parts[-1]

    print("=" * 20)

    save_path = HERE / "results" / "trial_arrays"
    redo = False

    if (save_path / f"{subject}_training_array.npy").exists() and not redo:
        print(f"Loading existing data for {subject}")
        training_array = np.load(save_path / f"{subject}_training_array.npy")
        testing_array = np.load(save_path / f"{subject}_testing_array.npy")
        training_labels = np.load(save_path / f"{subject}_training_labels.npy")
        testing_labels = np.load(save_path / f"{subject}_testing_labels.npy")
    else:
        training_arrays = []
        training_labels = []
        testing_arrays = []
        testing_labels = []

        for kilosort_path in kilosort_paths:

            ca1_low, ca1_high, rsc_low, rsc_high = get_ca1_rsc_channels(
                kilosort_path, df
            )

            print(f"CA1: {ca1_low} - {ca1_high}, RSC: {rsc_low} - {rsc_high}")

            train_array, y_train, test_array, y_test = process_probe(
                data_path, kilosort_path, (ca1_low, ca1_high, rsc_low, rsc_high)
            )
            training_arrays.append(train_array)
            training_labels.append(y_train)
            testing_arrays.append(test_array)
            testing_labels.append(y_test)

        training_array = np.concatenate(training_arrays, axis=1)
        testing_array = np.concatenate(testing_arrays, axis=1)
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

    if len(training_labels) > 1:
        assert np.array_equal(training_labels[0], training_labels[1])
        assert np.array_equal(testing_labels[0], testing_labels[1])

    model, label_encoder, awake_score, awake_shuffled_scores = get_awake_classifier(
        training_array.copy(),
        training_labels[0],
        C=C,
        penalty=penalty,
        solver=solver,
    )
    np.save(
        HERE / "results" / f"{subject}_awake_shuffled_scores.npy", awake_shuffled_scores
    )
    trial_states = get_sleep_state(data_path)

    assert len(trial_states) == testing_array.shape[0]
    trials_keep = np.isin(trial_states, ["nrem", "deep nrem"])

    print(
        f"Keeping {np.sum(trials_keep)} trials out of {len(trial_states)} based on sleep state."
    )

    testing_array = testing_array[trials_keep, :, :]
    testing_labels = testing_labels[0][trials_keep]
    assert (
        len(testing_labels) > 0
    ), "Testing labels are empty after filtering by sleep state."

    # score, shuffled_scores = fit_classifier_to_sleeping_data(
    #     testing_array.copy(), testing_labels, model, label_encoder, offset=offset
    # )

    # Optional: Test improved methods (set to True to compare methods)

    method_results = compare_classification_methods_rigorous(
        training_array,
        testing_array,
        training_labels[0],
        testing_labels,
        model,
        label_encoder,
        C,
        penalty,
        solver,
        offset=offset,
        subject=subject,
    )

    # score = method_results["cross_domain"]
    # score_shuffled_labels = method_results["cross_domain_shuffled_labels"]

    score = method_results["principled_threshold"]
    score_shuffled_labels = method_results["principled_threshold_shuffled_labels"]

    np.save(HERE / "results" / f"{subject}_classifier_score.npy", score)
    np.save(
        HERE / "results" / f"{subject}_classifier_shuffled_labels.npy",
        score_shuffled_labels,
    )
    np.save(HERE / "results" / f"{subject}_awake_scores.npy", awake_score)


def process_probe(
    data_path: Path,
    kilosort_path: Path,
    region_boundaries: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

    # assert sum(len(rsync) for rsync in rsync_times) == len(
    #     npx_sync_times
    # ), "Rsync times and NPX sync times do not match in length."
    aligners = get_aligners(npx_sync_times, rsync_times)
    spike_times, spike_clusters, labels, closest_channel = load_spiking_data(
        kilosort_path
    )

    idx_keep = (labels == "good") & (
        (closest_channel >= ca1_low) & (closest_channel <= ca1_high)
    ) | ((closest_channel >= rsc_low) & (closest_channel <= rsc_high))

    # idx_keep = ((closest_channel >= ca1_low) & (closest_channel <= ca1_high)) | (
    #     (closest_channel >= rsc_low) & (closest_channel <= rsc_high)
    # )

    # idx_keep = (closest_channel >= ca1_low) & (closest_channel <= ca1_high)
    # idx_keep = labels == "good"

    good = spike_times[idx_keep]
    spike_clusters = spike_clusters[idx_keep]
    print(f"Number of clusters : {len(np.unique(spike_clusters))}")

    n_bins = 100
    window = 0.9  # seconds
    train_array, y_train = get_training_data(
        sessions, aligners, spike_clusters, good, n_bins, window
    )

    test_array, y_test = get_testing_data(
        sessions, aligners, spike_clusters, good, n_bins, window
    )
    return train_array, y_train, test_array, y_test


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
    plot: bool = False,
    offset: int = 0,
) -> Tuple[float, List[float]]:

    X = X_from_trial_array(trial_array, offset=offset)

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
    score = np.sum(y_encoded == result) / len(y_encoded)

    print(f"Classifier score on sleeping data: {score:.2f}")

    shuffled_scores = []
    for _ in range(100):
        shuffle_idx = np.random.permutation(len(y_encoded))
        X_shuffled = X[shuffle_idx, :]
        result_shuffled = awake_model.predict(X_shuffled)
        shuffled_scores.append(np.sum(y_encoded == result_shuffled) / len(y_encoded))

    if plot:
        plt.figure()
        plt.hist(shuffled_scores, bins=15, alpha=0.5, label="Shuffled scores")
        plt.axvline(score, color="red", label="Original score")
        plt.title(
            f"Z = {(score - np.mean(shuffled_scores)) / np.std(shuffled_scores):.2f}"
        )

    # return score, shuffled_scores
    zscore = (score - np.mean(shuffled_scores)) / np.std(shuffled_scores)
    assert not np.isnan(zscore), "Z-score is NaN. Likely only one label is predicted"
    return score, shuffled_scores


def get_testing_data(
    sessions: List[Session],
    aligners: List[Rsync_aligner],
    spike_clusters: np.ndarray,
    good: np.ndarray,
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
        spikes=good,
        spike_clusters=spike_clusters,
        window=window,
        n_bins=n_bins,
    )

    return trial_array, np.array([sound.frequency for sound in sounds])


def X_from_trial_array(trial_array: np.ndarray, offset: int = 0) -> np.ndarray:
    # (n_trials, n_clusters, n_bins)
    # Transform to (n_samples, n_features)
    normalize_trial_array(trial_array)
    n_bins = trial_array.shape[2]
    start = (n_bins // 2) + offset
    end = start + 10
    X = trial_array[:, :, start:end]
    # return X.reshape(X.shape[0], -1)
    return X.mean(axis=2)


def get_awake_classifier(
    trial_array: np.ndarray, y: np.ndarray, C: float, penalty: str, solver: str
) -> Tuple[LogisticRegression, LabelEncoder, float]:

    X = X_from_trial_array(trial_array)

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

    shuffled_scores = []
    for _ in range(100):
        shuffle_idx = np.random.permutation(len(y_encoded))
        y_shuffled = y_encoded[shuffle_idx]
        shuffled_score = cross_val_score(
            model, X, y_shuffled, cv=cv, scoring="accuracy"
        )
        shuffled_scores.append(shuffled_score.mean())

    return model, le, scores.mean(), shuffled_scores


def get_training_data(
    sessions: List[Session],
    aligners: List[Rsync_aligner],
    spike_clusters: np.ndarray,
    good: np.ndarray,
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
        spikes=good,
        spike_clusters=spike_clusters,
        window=window,
        n_bins=n_bins,
    )

    return (
        trial_array,
        colors,
    )


def plot_processed_data() -> float:
    # results_path = Path(__file__).parent / "results"
    results_path = Path("/Volumes/MarcBusche/James/Alex/alex_results_working")

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
    plt.ylim(None, 0.75)
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
    # results_path = Path(__file__).parent / "results"
    results_path = Path("/Volumes/MarcBusche/James/Alex/alex_results_working")
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

    umbrella = Path("/Volumes/MarcBusche/Alex/Reactivations")

    df = gsheet2df("112rq_5qilRHtYUFnFwpjDQeF4XKyTdY6qJhIwAnykN8", "Sheet1", 1)

    kilosort_paths = list(umbrella.rglob("*/kilosort4"))
    path_dict: Dict[str, List[Path]] = {}
    if len(kilosort_paths) == 0:
        raise FileNotFoundError(
            "No kilosort paths found. Please check the path to the data."
        )
    for kilosort_path in kilosort_paths:
        mouse = kilosort_path.parts[-4]
        # This mouse has bad data, see email
        if mouse in ["11153"]:
            continue
        if mouse not in path_dict:
            path_dict[mouse] = []
        path_dict[mouse].append(kilosort_path)

    solver = "liblinear"  # Solver for Logistic Regression
    penalty = "l1"

    for C in [5]:

        for mouse, kilosort_paths in path_dict.items():

            assert all(
                [
                    (kilosort_path / "spike_times.npy").exists()
                    for kilosort_path in kilosort_paths
                ]
            )

            data_path = kilosort_paths[0].parent.parent.parent
            process_mouse(data_path, df, kilosort_paths, C, penalty, solver, offset=0)

        # awake_score = plot_processed_data()
        # plt.title(
        #     f"C = {}, penalty = {penalty}, solver = {solver}, awake = {awake_score: .2f} offset = {0}"
        # )

    plt.show()


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


def apply_threshold_correction(
    trial_array: np.ndarray,
    y: np.ndarray,
    awake_model: LogisticRegression,
    label_encoder: LabelEncoder,
    offset: int = 0,
) -> Tuple[float, List[float]]:
    """Apply threshold correction to compensate for decision boundary shift."""

    print("\n" + "=" * 30 + " THRESHOLD CORRECTION " + "=" * 30)

    X = X_from_trial_array(trial_array.copy(), offset=offset)
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

    # Get decision scores
    decision_scores = awake_model.decision_function(X)

    # Find optimal threshold by maximizing accuracy
    thresholds = np.linspace(decision_scores.min(), decision_scores.max(), 100)
    accuracies = []

    for threshold in thresholds:
        predictions = (decision_scores > threshold).astype(int)
        accuracy = np.mean(predictions == y_encoded)
        accuracies.append(accuracy)

    optimal_threshold = thresholds[np.argmax(accuracies)]
    optimal_accuracy = max(accuracies)

    print(f"Original threshold: 0.0000")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Original accuracy: {np.mean((decision_scores > 0) == y_encoded):.4f}")
    print(f"Corrected accuracy: {optimal_accuracy:.4f}")
    print(
        f"Improvement: {optimal_accuracy - np.mean((decision_scores > 0) == y_encoded):.4f}"
    )

    # Apply corrected threshold
    corrected_predictions = (decision_scores > optimal_threshold).astype(int)

    print(
        f"Corrected predictions: blue={np.sum(corrected_predictions == 0)}, orange={np.sum(corrected_predictions == 1)}"
    )
    print(
        f"True labels: blue={np.sum(y_encoded == 0)}, orange={np.sum(y_encoded == 1)}"
    )

    score = optimal_accuracy

    # Calculate z-score with shuffled data
    shuffled_scores = []
    for _ in range(100):
        shuffle_idx = np.random.permutation(len(y_encoded))
        X_shuffled = X[shuffle_idx, :]
        scores_shuffled = awake_model.decision_function(X_shuffled)
        pred_shuffled = (scores_shuffled > optimal_threshold).astype(int)
        shuffled_scores.append(np.mean(pred_shuffled == y_encoded))

    zscore = (score - np.mean(shuffled_scores)) / np.std(shuffled_scores)

    print(f"Threshold-corrected Z-score: {zscore:.3f}")
    print("=" * 75)

    return zscore, shuffled_scores


def apply_principled_threshold_correction(
    trial_array: np.ndarray,
    y: np.ndarray,
    awake_model: LogisticRegression,
    label_encoder: LabelEncoder,
    offset: int = 0,
    validation_split: float = 0.5,
    verbose: bool = False,
) -> Tuple[float, List[float], float]:
    """Apply threshold correction with proper validation to avoid data snooping."""

    X = X_from_trial_array(trial_array.copy(), offset=offset)
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

    # Split data: use part for threshold optimization, part for evaluation
    n_samples = len(y_encoded)
    split_idx = int(n_samples * validation_split)

    # Randomly split (but reproducibly)
    np.random.seed(42)
    shuffle_idx = np.random.permutation(n_samples)

    # Optimization set (find threshold)
    opt_idx = shuffle_idx[:split_idx]
    X_opt, y_opt = X[opt_idx], y_encoded[opt_idx]

    # Evaluation set (test threshold)
    eval_idx = shuffle_idx[split_idx:]
    X_eval, y_eval = X[eval_idx], y_encoded[eval_idx]

    if verbose:
        print(f"Using {len(opt_idx)} trials for threshold optimization")
        print(f"Using {len(eval_idx)} trials for evaluation")

    # Find optimal threshold on optimization set
    decision_scores_opt = awake_model.decision_function(X_opt)

    thresholds = np.linspace(
        decision_scores_opt.min(),
        decision_scores_opt.max(),
        100,
    )
    accuracies = []

    for threshold in thresholds:
        predictions = (decision_scores_opt > threshold).astype(int)
        # Use balanced accuracy score instead of simple accuracy

        accuracy = balanced_accuracy_score(y_opt, predictions)
        # accuracy = np.mean(predictions == y_opt)
        accuracies.append(accuracy)

    optimal_threshold = thresholds[np.argmax(accuracies)]

    if verbose:
        print(f"Optimal threshold found: {optimal_threshold:.4f}")

    # Apply threshold to evaluation set (unseen during optimization)
    decision_scores_eval = awake_model.decision_function(X_eval)
    corrected_predictions = (decision_scores_eval > optimal_threshold).astype(int)

    score = balanced_accuracy_score(y_eval, corrected_predictions)
    if verbose:
        print(f"principle threshold score: {score:.4f}")
    # assert not np.all(
    #     corrected_predictions == corrected_predictions[0]
    # ), "All predictions are the same. This indicates a problem with the threshold or the data."

    original_score = balanced_accuracy_score(
        y_eval, (decision_scores_eval > 0).astype(int)
    )
    return score, None, None
    print(f"Number of trials in eval set: {len(y_eval)}")

    print(f"Original accuracy on eval set: {original_score:.4f}")
    print(f"Corrected accuracy on eval set: {score:.4f}")
    print(f"Improvement: {score - original_score:.4f}")

    # # Calculate z-score with shuffled data (on evaluation set only)
    # shuffled_scores = []
    # for _ in range(1000):
    #     shuffle_eval_idx = np.random.permutation(len(y_eval))
    #     X_shuffled = X_eval[shuffle_eval_idx, :]
    #     scores_shuffled = awake_model.decision_function(X_shuffled)
    #     pred_shuffled = (scores_shuffled > optimal_threshold).astype(int)
    #     shuffled_scores.append(np.mean(pred_shuffled == y_eval))

    # # assert not np.allclose(
    # #     shuffled_scores, shuffled_scores[0], atol=1e-6
    # # ), "Shuffled scores are all the same. This indicates a problem with the shuffling or the data."
    # zscore = (score - np.mean(shuffled_scores)) / np.std(shuffled_scores)
    # # assert not np.isnan(zscore), "Z-score is NaN. Likely only one label is predicted."

    # print(f"Principled threshold-corrected Z-score: {zscore:.3f}")
    # print("=" * 75)

    return zscore, shuffled_scores, optimal_threshold


def apply_cross_domain_normalization(
    training_array: np.ndarray,
    testing_array: np.ndarray,
    training_labels: np.ndarray,
    testing_labels: np.ndarray,
    C: float,
    penalty: str,
    solver: str,
    offset: int = 0,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, LogisticRegression, LabelEncoder]:
    """
    Apply cross-domain normalization: normalize features across both domains,
    retrain model, and evaluate on normalized test data.

    This addresses domain shift by ensuring both training and testing data
    have similar feature distributions.
    """

    # Extract features from both domains
    X_train = X_from_trial_array(training_array.copy(), offset=offset)
    X_test = X_from_trial_array(testing_array.copy(), offset=offset)

    if verbose:
        print(f"Original training data shape: {X_train.shape}")
        print(f"Original testing data shape: {X_test.shape}")
        print(f"Training mean activity: {X_train.mean():.4f}")
        print(f"Testing mean activity: {X_test.mean():.4f}")

    # Combine data for normalization statistics
    X_combined = np.vstack([X_train, X_test])

    # Calculate normalization parameters from combined data
    combined_mean = X_combined.mean(axis=0)
    combined_std = X_combined.std(axis=0)
    combined_std[combined_std == 0] = 1  # Avoid division by zero

    # Normalize both training and testing data using combined statistics
    X_train_norm = (X_train - combined_mean) / combined_std
    X_test_norm = (X_test - combined_mean) / combined_std
    # Encode labels
    label_encoder_new = LabelEncoder()
    y_train_encoded = label_encoder_new.fit_transform(training_labels)

    blue_encoding, orange_encoding = label_encoder_new.transform(["blue", "orange"])
    y_test_encoded = np.array(
        [
            (
                blue_encoding
                if sound == 3000
                else orange_encoding if sound == 8000 else None
            )
            for sound in testing_labels
        ]
    )

    if verbose:
        print(f"Training labels: {len(y_train_encoded)} samples")
        print(f"Testing labels: {len(y_test_encoded)} samples")

    # Train new model on normalized training data
    model_norm = LogisticRegression(
        C=C, penalty=penalty, solver=solver, random_state=42, max_iter=1000
    )
    model_norm.fit(X_train_norm, y_train_encoded)

    # Evaluate on normalized testing data
    predictions = model_norm.predict(X_test_norm)

    assert not np.all(
        predictions == predictions[0]
    ), "All predictions are the same. This indicates a problem with the normalization or the data."

    score = balanced_accuracy_score(y_test_encoded, predictions)

    if verbose:
        print(f"cross norm score: {score}")
    return score, None, None, None

    if verbose:
        print(f"Cross-domain normalized accuracy: {score:.4f}")

    # Calculate z-score with shuffled data (using normalized features)
    shuffled_scores = []
    for _ in range(1000):
        # Shuffle the testing data
        shuffle_idx = np.random.permutation(len(y_test_encoded))
        X_test_shuffled = X_test_norm[shuffle_idx, :]
        pred_shuffled = model_norm.predict(X_test_shuffled)
        shuffled_scores.append(balanced_accuracy_score(y_test_encoded, pred_shuffled))

    zscore = (score - np.mean(shuffled_scores)) / np.std(shuffled_scores)

    if verbose:
        print(f"Cross-domain normalized Z-score: {zscore:.3f}")
        print(f"Shuffled mean: {np.mean(shuffled_scores):.4f}")
        print(f"Shuffled std: {np.std(shuffled_scores):.4f}")
        print("=" * 75)

    return score, shuffled_scores, model_norm, label_encoder_new


def justify_threshold_correction_scientifically(
    training_array: np.ndarray,
    testing_array: np.ndarray,
    model: LogisticRegression,
    label_encoder: LabelEncoder,
) -> None:
    """Provide scientific justification for why threshold correction might be needed."""

    print("\n" + "=" * 30 + " SCIENTIFIC JUSTIFICATION " + "=" * 30)

    # Extract features
    X_train = X_from_trial_array(training_array.copy())
    X_test = X_from_trial_array(testing_array.copy())

    # Compare feature distributions
    print("FEATURE DISTRIBUTION ANALYSIS:")
    print(f"Training (awake) mean activity: {X_train.mean():.4f}")
    print(f"Testing (sleep) mean activity: {X_test.mean():.4f}")
    print(f"Activity shift: {X_test.mean() - X_train.mean():.4f}")

    # Decision boundary analysis
    train_scores = model.decision_function(X_train)
    test_scores = model.decision_function(X_test)

    print(f"\nDECISION BOUNDARY ANALYSIS:")
    print(
        f"Training decision scores: {train_scores.mean():.4f} ± {train_scores.std():.4f}"
    )
    print(
        f"Testing decision scores: {test_scores.mean():.4f} ± {test_scores.std():.4f}"
    )
    print(f"Boundary shift: {test_scores.mean() - train_scores.mean():.4f}")

    # Theoretical justification
    boundary_shift = abs(test_scores.mean() - train_scores.mean())
    activity_shift = abs(X_test.mean() - X_train.mean())

    print(f"\nSCIENTIFIC REASONING:")
    if boundary_shift > 0.5:
        print("✓ Large decision boundary shift detected")
        print("  → Suggests systematic difference between sleep/wake states")
        print("  → Threshold correction addresses domain shift, not signal quality")

    if activity_shift > 0.1:
        print("✓ Substantial activity level difference between states")
        print("  → Sleep suppression is expected neurobiologically")
        print("  → Correction accounts for state-dependent baseline shifts")

    print(f"\nJUSTIFICATION VERDICT:")
    if boundary_shift > 0.5 and activity_shift > 0.1:
        print("🟢 THRESHOLD CORRECTION IS SCIENTIFICALLY JUSTIFIED")
        print("   Reason: Correcting for known neurobiological differences")
        print("   between sleep and wake states, not optimizing signal detection")
    else:
        print("🟡 THRESHOLD CORRECTION IS QUESTIONABLE")
        print("   Reason: Changes may reflect genuine differences in reactivation")

    print("=" * 75)


# Add this to your comparison function
def compare_classification_methods_rigorous(
    training_array: np.ndarray,
    testing_array: np.ndarray,
    training_labels: np.ndarray,
    testing_labels: np.ndarray,
    model: LogisticRegression,
    label_encoder: LabelEncoder,
    C: float,
    penalty: str,
    solver: str,
    offset: int = 0,
    subject: str = "Unknown",
) -> Dict[str, float]:

    results = {}

    # 1. Original method
    # print("\n1. ORIGINAL METHOD:")
    # original_zscore, _ = fit_classifier_to_sleeping_data(
    #     testing_array.copy(), testing_labels, model, label_encoder, offset=offset
    # )
    # results["original"] = original_zscore

    # 2. Scientific justification for threshold correction
    # justify_threshold_correction_scientifically(
    #     training_array, testing_array, model, label_encoder
    # )

    # 3. Principled threshold correction (with validation split)
    # print("\n2. PRINCIPLED THRESHOLD CORRECTION:")

    principled_zscore, _, threshold = apply_principled_threshold_correction(
        testing_array.copy(),
        testing_labels,
        model,
        label_encoder,
        offset=offset,
        verbose=True,
    )

    results["principled_threshold"] = principled_zscore

    shuffled_result = []
    for _ in range(100):
        np.random.shuffle(testing_labels)
        principle_zscore_shuffled, _, threshold = apply_principled_threshold_correction(
            testing_array.copy(), testing_labels, model, label_encoder, offset=offset
        )
        shuffled_result.append(principle_zscore_shuffled)

    results["principled_threshold_shuffled_labels"] = shuffled_result

    # 4. Cross-domain normalization

    ###############
    print("\n3. CROSS-DOMAIN NORMALIZATION:")
    cross_norm_score, _, _, _ = apply_cross_domain_normalization(
        training_array.copy(),
        testing_array.copy(),
        training_labels,
        testing_labels,
        C,
        penalty,
        solver,
        offset=offset,
        verbose=True,
    )

    shuffled_result = []

    for _ in range(100):
        np.random.shuffle(testing_labels)
        domain_zscore_shuffled_labels, _, _, _ = apply_cross_domain_normalization(
            training_array.copy(),
            testing_array.copy(),
            training_labels,
            testing_labels,
            C,
            penalty,
            solver,
            offset=offset,
        )
        shuffled_result.append(domain_zscore_shuffled_labels)

    # results["cross_domain"] = cross_norm_score
    # results["cross_domain_shuffled_labels"] = shuffled_result

    return results


if __name__ == "__main__":
    # main()
    plot_processed_data()
    plot_processed_data_waking()
    plt.show()
