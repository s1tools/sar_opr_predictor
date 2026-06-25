import numpy as np
import pytest

from s1tools.opr_predictor.opr_inference import CONFIG as OPR_CONFIG, block_mean_downsample


FACTOR: int = int(round(OPR_CONFIG.output_resolution_m / OPR_CONFIG.target_resolution_m))


@pytest.fixture(scope="module")
def random_ground_mask() -> np.ndarray:
    rng = np.random.default_rng(seed=20230413)
    return (rng.random((40, 60)) < 0.3).astype(np.float32)


def test_factor_is_five() -> None:
    assert FACTOR == 5


def test_downsample_shape(random_ground_mask: np.ndarray) -> None:
    out = block_mean_downsample(random_ground_mask, FACTOR)
    assert out.shape == (random_ground_mask.shape[0] // FACTOR, random_ground_mask.shape[1] // FACTOR)


def test_downsample_all_zero() -> None:
    out = block_mean_downsample(np.zeros((40, 60), dtype=np.float32), FACTOR)
    assert np.all(out == 0)


def test_downsample_all_one() -> None:
    out = block_mean_downsample(np.ones((40, 60), dtype=np.float32), FACTOR)
    assert np.all(out == 1)


def test_downsample_nan_aware() -> None:
    array = np.ones((40, 60), dtype=np.float32)
    array[::2, ::2] = np.nan
    out = block_mean_downsample(array, FACTOR)
    assert np.all(np.isfinite(out))
    assert np.all(out == 1)


def test_ground_mask_consistency(random_ground_mask: np.ndarray) -> None:
    ny, nx = random_ground_mask.shape
    reshaped = random_ground_mask.reshape(ny // FACTOR, FACTOR, nx // FACTOR, FACTOR)
    direct_1km = (reshaped.sum(axis=(1, 3)) >= (FACTOR * FACTOR) / 2.0).astype(np.float32)
    downsampled_thresholded = (block_mean_downsample(random_ground_mask, FACTOR) >= 0.5).astype(np.float32)
    np.testing.assert_array_equal(downsampled_thresholded, direct_1km)


def test_ground_mask_homogeneous_blocks() -> None:
    array = np.zeros((20, 30), dtype=np.float32)
    array[:5, :5] = 1.0
    array[5:10, 10:15] = 1.0
    out = block_mean_downsample(array, FACTOR)
    assert out[0, 0] == 1.0
    assert out[1, 2] == 1.0
    expected_zero_mask = np.ones_like(out, dtype=bool)
    expected_zero_mask[0, 0] = False
    expected_zero_mask[1, 2] = False
    assert np.all(out[expected_zero_mask] == 0.0)
