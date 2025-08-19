import numpy as np

from lfp_signatures import get_baseline_spindles


def test_get_baseline_spindles() -> None:
    spindle_times = np.array([100, 160, 300])
    ripple_times = np.array([150, 250])
    ripple_distance = 20

    baseline_spindles = get_baseline_spindles(
        spindle_times, ripple_times, ripple_distance
    )

    expected_baseline_spindles = np.array([100, 300])
    np.testing.assert_array_equal(baseline_spindles, expected_baseline_spindles)


def test_get_baseline_spindles2() -> None:

    spindle_times = np.array([100, 250, 300])
    ripple_times = np.array([1, 10000])
    ripple_distance = 20

    baseline_spindles = get_baseline_spindles(
        spindle_times, ripple_times, ripple_distance
    )

    expected_baseline_spindles = np.array([100, 250, 300])
    np.testing.assert_array_equal(baseline_spindles, expected_baseline_spindles)


def test_get_baseline_spindles3() -> None:

    spindle_times = np.array([101, 250, 300])
    ripple_times = np.array([100, 102, 103, 250, 249, 300, 301])
    ripple_distance = 5

    baseline_spindles = get_baseline_spindles(
        spindle_times, ripple_times, ripple_distance
    )

    expected_baseline_spindles = np.array([])
    np.testing.assert_array_equal(baseline_spindles, expected_baseline_spindles)
