import numpy as np

from utils import zero_same_region


def test_zero_same_region_all_same() -> None:
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    regions = np.array(["A", "A", "A"])
    expected_output = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.array_equal(zero_same_region(mat, regions), expected_output)


def test_zero_same_region() -> None:
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    regions = np.array(["A", "B", "A"])
    expected_output = np.array([[0, 2, 0], [4, 0, 6], [0, 8, 0]])
    result = zero_same_region(mat, regions)
    assert np.array_equal(result, expected_output)


def test_zero_same_region2() -> None:
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    regions = np.array(["A", "B", "C"])
    result = zero_same_region(mat.copy(), regions)
    np.fill_diagonal(mat, 0)
    assert np.array_equal(result, mat)
