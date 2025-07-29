from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Tuple

from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from data_import import Session
from scipy.stats import zscore
from main import process_session
from rsync import Rsync_aligner
from utils import get_aligners, get_data_paths

from ripples.utils_npyx import load_sync_npyx
from ripples.utils import threshold_detect
from plotting import plot_cell_summed_results

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif


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

    channel_depth = np.load(kilosort_path / "channel_positions.npy")[:, 1]
    spike_depth = np.load(kilosort_path / "spike_positions.npy")[:, 1]
    # Find the closest channel depth for each spike in a memory-efficient way

    if (kilosort_path / "closest_channel.npy").exists():
        closest_channel = np.load(kilosort_path / "closest_channel.npy")
    else:
        closest_channel = np.array(
            [np.argmin(np.abs(channel_depth - sd)) for sd in spike_depth]
        )
        np.save(kilosort_path / "closest_channel.npy", closest_channel)

    return spike_times, spike_clusters, label_array, closest_channel


def split_data_by_trial(
    stim_times: np.ndarray,
    spikes: np.ndarray,
    spike_clusters: np.ndarray,
    window: float,
    n_bins: int,
) -> np.ndarray:
    """Split the spike data into trials based on stimulus times.

    Returns: np.ndarray (n_trials, n_clusters, n_bins)
    """

    result = []
    sampling_rate = 30_000
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


def get_ca1_rsc_mapping(kilosort_path: Path) -> Tuple[int, int, int, int]:

    mapping_df = pd.read_csv(
        "/Users/jamesrowland/Desktop/mouse_lfp_mapping_full_population.csv"
    )
    found = False
    for _, row in mapping_df.iterrows():
        p = PureWindowsPath(row["lfp_path"])
        if (
            p.parts[-1] == kilosort_path.parts[-2]
            and p.parts[-2] == kilosort_path.parts[-3]
        ):
            assert not found, "Found multiple mappings for the same kilosort path."
            ca1_low = row["CA1_Low"]
            ca1_high = row["CA1_High"]
            rsc_low = row["RSC_Low"]
            rsc_high = row["RSC_High"]
            found = True

    if found:
        return ca1_low, ca1_high, rsc_low, rsc_high

    raise ValueError(f"Could not find mapping for kilosort path {kilosort_path}. ")


def main(data_path: Path, kilosort_path: Path) -> None:

    CA1_Low, CA1_High, RSC_Low, RSC_High = get_ca1_rsc_mapping(kilosort_path)
    print(f"CA1: {CA1_Low} - {CA1_High}, RSC: {RSC_Low} - {RSC_High}")

    _, _, pycontrol_files = get_data_paths(data_path)
    sessions = [Session(pycontrol_file) for pycontrol_file in pycontrol_files]

    rsync_times = [session.times["rsync"] for session in sessions]

    if (kilosort_path.parent / "high_pass_sync.npy").exists():
        print("Loading existing sync file")
        npx_sync_times = np.load(kilosort_path.parent / "high_pass_sync.npy")
    else:
        sync = load_sync_npyx(str(kilosort_path.parent), filt_key="highpass")
        npx_sync_times = threshold_detect(sync, 0.5)
        np.save(kilosort_path.parent / "high_pass_sync.npy", npx_sync_times)

    assert sum(len(rsync) for rsync in rsync_times) == len(
        npx_sync_times
    ), "Rsync times and NPX sync times do not match in length."
    aligners = get_aligners(npx_sync_times, rsync_times)
    spike_times, spike_clusters, labels, closest_channel = load_spiking_data(
        kilosort_path
    )

    idx_keep = (labels == "good") & (
        (closest_channel >= CA1_Low) & (closest_channel <= CA1_High)
    ) | ((closest_channel >= RSC_Low) & (closest_channel <= RSC_High))

    # idx_keep = ((closest_channel >= CA1_Low) & (closest_channel <= CA1_High)) | (
    #     (closest_channel >= RSC_Low) & (closest_channel <= RSC_High)
    # )

    # idx_keep = labels == "good"

    good = spike_times[idx_keep]
    spike_clusters = spike_clusters[idx_keep]
    print(f"Number of clusters : {len(np.unique(spike_clusters))}")

    n_bins = 101
    window = 0.1  # seconds

    model, label_encoder = get_awake_classifier(
        sessions,
        aligners,
        spike_clusters,
        good,
        n_bins=n_bins,
        window=window,
    )

    fit_classifier_to_sleeping_data(
        sessions,
        aligners,
        spike_clusters,
        label_encoder,
        model,
        good,
        n_bins=n_bins,
        window=window,
    )


def fit_classifier_to_sleeping_data(
    sessions: List[Session],
    aligners: List[Rsync_aligner],
    spike_clusters: np.ndarray,
    label_encoder: LabelEncoder,
    awake_model: LogisticRegression,
    good: np.ndarray,
    n_bins: int = 51,
    window: float = 0.5,
) -> None:

    session_index = 2

    sounds, _ = process_session(sessions[session_index])

    sounds_times = aligners[session_index].B_to_A(
        np.array([sound.time for sound in sounds]), extrapolate=False
    )

    trial_array = split_data_by_trial(
        stim_times=sounds_times,
        spikes=good,
        spike_clusters=spike_clusters,
        window=window,
        n_bins=n_bins,
    )

    # (n_trials, n_clusters, n_bins)
    # Transform to (n_samples, n_features)
    # Sum across the post stimulus bins
    X = trial_array[:, :, n_bins // 2 :].sum(axis=2)

    # Blue -> 3000, Orange -> 8000

    blue_encoding, orange_encoding = label_encoder.transform(["blue", "orange"])

    y = np.array(
        [
            (
                blue_encoding
                if sound.frequency == 3000
                else orange_encoding if sound.frequency == 8000 else None
            )
            for sound in sounds
        ]
    )
    assert np.all(y is not None), "Some sounds do not have a valid frequency encoding."

    result = awake_model.predict(X)
    score = np.sum(y == result) / len(y)
    print(f"Classifier score on sleeping data: {score:.2f}")

    shuffled_scores = []
    for _ in range(1000):
        shuffle_idx = np.random.permutation(len(y))
        X_shuffled = X[shuffle_idx, :]
        result_shuffled = awake_model.predict(X_shuffled)
        shuffled_scores.append(np.sum(y == result_shuffled) / len(y))

    plt.hist(shuffled_scores, bins=15, alpha=0.5, label="Shuffled scores")
    plt.axvline(score, color="red", label="Original score")
    plt.title(f"Z = {(score - np.mean(shuffled_scores)) / np.std(shuffled_scores):.2f}")

    subject = sessions[0].subject_ID
    here = Path(__file__).parent
    if (here / "results" / f"{subject}_classifier_score.npy").exists():
        subject += "_1"
    np.save(here / "results" / f"{subject}_classifier_score.npy", score)
    np.save(here / "results" / f"{subject}_shuffled_scores.npy", shuffled_scores)


def get_awake_classifier(
    sessions: List[Session],
    aligners: List[Rsync_aligner],
    spike_clusters: np.ndarray,
    good: np.ndarray,
    n_bins: int = 51,
    window: float = 0.5,
) -> Tuple[LogisticRegression, LabelEncoder]:

    session_index = 0

    _, LEDs = process_session(sessions[session_index])
    assert LEDs is not None

    LEDs_times = aligners[session_index].B_to_A(
        np.array([led.time for led in LEDs]), extrapolate=False
    )

    trial_array = split_data_by_trial(
        stim_times=LEDs_times,
        spikes=good,
        spike_clusters=spike_clusters,
        window=window,
        n_bins=n_bins,
    )

    # (n_trials, n_clusters, n_bins)
    # Transform to (n_samples, n_features)
    # Sum across the post stimulus bins

    X = trial_array[:, :, n_bins // 2 :].sum(axis=2)
    y = np.array([led.color for led in LEDs])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    C = 0.1  # Regularization strength for Logistic Regression
    solver = "liblinear"  # Solver for Logistic Regression
    penalty = "l1"
    model = LogisticRegression(penalty=penalty, solver=solver, C=C)

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy")

    print("Mean 5-fold accuracy:", scores.mean())
    np.save(Path(__file__).parent / "results" / "awake_classifier_scores.npy", scores)

    model = LogisticRegression(penalty=penalty, solver=solver, C=C)

    model.fit(X, y_encoded)
    return model, le


def plot_processed_data() -> None:
    results_path = Path(__file__).parent / "results"
    mice = set([file.stem.split("_")[0] for file in results_path.glob("*.npy")])
    WT_scores = []
    NLGF_scores = []

    for mouse in mice:
        score1 = np.load(results_path / f"{mouse}_classifier_score.npy").item()
        score2_path = results_path / f"{mouse}_1_classifier_score.npy"
        if not score2_path.exists():
            print(
                f"Warning: {score2_path} does not exist, using only {mouse}_classifier_score.npy"
            )

        total_score = (
            np.mean([score1, np.load(score2_path).item()])
            if score2_path.exists()
            else score1
        )
        if mouse[:3] == "000":
            WT_scores.append(total_score)
        else:
            NLGF_scores.append(total_score)

    # WT_scores = []
    # NLGF_scores = []
    # WT_shuffles = []
    # NLGF_shuffles = []
    # for file in results_path.glob("*.npy"):
    #     genotype = "WT" if file.stem.split("_")[0][:3] == "000" else "NLGF/S305N"
    #     if "classifier_score" in file.stem and "awake" not in file.stem:
    #         score = np.load(file)
    #         if genotype == "WT":
    #             WT_scores.append(score.item())
    #         else:
    #             NLGF_scores.append(score.item())
    #     if "shuffled_scores" in file.stem and "awake" not in file.stem:
    #         shuffled_scores = np.load(file)
    #         if genotype == "WT":
    #             WT_shuffles.extend(shuffled_scores)
    #         else:
    #             NLGF_shuffles.extend(shuffled_scores)

    sns.boxplot(data={"WT": WT_scores, "NLGF/S305N": NLGF_scores}, showfliers=False)
    plt.title(
        f"Classifier scores by genotype (p = {ttest_ind(WT_scores, NLGF_scores).pvalue:.3f})"
    )
    plt.axhline(0.5, color="red", linestyle="--", label="Chance level")
    plt.legend()
    # plt.show()
    # plt.figure()
    # plt.hist(
    #     WT_shuffles, bins=15, alpha=0.5, label="WT shuffled", color="blue", density=True
    # )
    # plt.axvline(np.mean(WT_scores), color="red", linestyle="--", label="WT mean")

    # plt.figure()

    # plt.hist(
    #     NLGF_shuffles,
    #     bins=15,
    #     alpha=0.5,
    #     label="WT shuffled",
    #     color="blue",
    #     density=True,
    # )
    # plt.axvline(np.mean(NLGF_scores), color="red", linestyle="--", label="NLGF mean")
    1 / 0


if __name__ == "__main__":
    plot_processed_data()

    # umbrella = Path("/Volumes/MarcBusche/Alex/Reactivations")

    # for kilosort_path in umbrella.rglob("*/kilosort4"):
    #     if not (kilosort_path / "spike_times.npy").exists():
    #         print(f"Skipping {kilosort_path} as it does not contain spike_times.npy")
    #         continue
    #     print(f"Processing {kilosort_path}")
    #     data_path = kilosort_path.parent.parent.parent

    #     try:
    #         main(data_path, kilosort_path)
    #     except Exception as e:
    #         print(f"Error processing {kilosort_path}: {e}")
    #         continue
