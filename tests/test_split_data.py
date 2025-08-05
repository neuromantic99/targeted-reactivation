import sys
import numpy as np
from pathlib import Path

here = Path(__file__).parent
sys.path.append(str(here.parent))
from reactivation_classifier import split_data_by_trial


def test_split_data_by_trial_basic() -> None:
    spikes = np.array([1, 2, 3, 4, 5, 6])
    spike_clusters = np.array([0] * 6)

    stim_times = np.array([3.5])  # One stimulus
    window = 4
    n_bins = 2

    sampling_rate = 1

    result = split_data_by_trial(
        stim_times, spikes, spike_clusters, window, n_bins, sampling_rate
    )

    # Result should have shape: (n_stims, max_cluster+1, n_bins-1)
    assert result.shape == (1, 1, n_bins)
    assert np.array_equal(result, np.array([[[3, 3]]]))


def test_split_data_by_trial_multiple_clusters() -> None:
    spikes = np.array([1, 2, 3, 4, 5, 6])
    spike_clusters = np.array([0, 0, 0, 1, 1, 1])

    stim_times = np.array([3.5])
    window = 4
    n_bins = 2

    sampling_rate = 1

    result = split_data_by_trial(
        stim_times, spikes, spike_clusters, window, n_bins, sampling_rate
    )

    # Result should have shape: (n_stims, max_cluster+1, n_bins-1)
    assert result.shape == (1, len(set(spike_clusters)), n_bins)
    expected = np.array([[[3, 0], [0, 3]]])
    assert np.array_equal(result, expected)

    # Make sure switchin the order of the clusters does not change the result
    spikes = np.array([4, 5, 6, 1, 2, 3])
    spike_clusters = np.array([1, 1, 1, 0, 0, 0])


def test_split_data_by_trial_multiple_clusters_multiple_trials() -> None:
    spikes = np.array([1.1, 2, 2.9, 4, 5.1, 6])
    spike_clusters = np.array([0, 0, 0, 1, 1, 1])

    stim_times = np.array([3, 11])
    window = 2
    n_bins = 2

    sampling_rate = 1

    bin_edges = np.linspace(-window * sampling_rate, window * sampling_rate, n_bins)
    # bin edges = -2 -> 0 and 0 -> 2

    # First trial = 1 - 3 and 3 - 5
    # Second trial = 9 - 11 and 11 - 13

    result = split_data_by_trial(
        stim_times, spikes, spike_clusters, window, n_bins, sampling_rate
    )

    expected = np.array(
        [
            [[3, 0], [0, 1]],  # First trial
            [[0, 0], [0, 0]],  # Second trial
        ]
    )

    # Result should have shape: (n_stims, max_cluster+1, n_bins-1)
    assert result.shape == (len(stim_times), len(set(spike_clusters)), n_bins)
    assert np.array_equal(result, expected)

    # Make sure switchin the order of the clusters does not change the result
    spikes = np.array([4, 5, 6, 1, 2, 3])
    spike_clusters = np.array([1, 1, 1, 0, 0, 0])


def test_split_data_by_trial_multiple_clusters() -> None:
    spikes = np.array([1, 2, 3, 4, 5, 6])
    spike_clusters = np.array([0, 0, 0, 1, 1, 1])

    stim_times = np.array([3.5])
    window = 4
    n_bins = 2

    sampling_rate = 1

    result = split_data_by_trial(
        stim_times, spikes, spike_clusters, window, n_bins, sampling_rate
    )

    # Result should have shape: (n_stims, max_cluster+1, n_bins-1)
    assert result.shape == (1, len(set(spike_clusters)), n_bins)
    expected = np.array([[[3, 0], [0, 3]]])
    assert np.array_equal(result, expected)

    # Make sure switchin the order of the clusters does not change the result
    spikes = np.array([4, 5, 6, 1, 2, 3])
    spike_clusters = np.array([1, 1, 1, 0, 0, 0])
