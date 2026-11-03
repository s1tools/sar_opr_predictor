import typing

import numpy as np
import pytest

from s1tools.opr_predictor.opr_inference import (
    OPRInference,
    fold,
    footprint_reflection_indices,
    get_inputs,
    line_limits,
)


SHAPE: typing.Tuple[int, int] = (833, 1328)
LAND = np.s_[400:440, 600:640]
BURST_STEP = 12


def staircase_footprint() -> np.ndarray:
    """Three overlapping subswaths of nine bursts each, skewed so that the outer edges are staircases."""
    footprint = np.zeros(SHAPE, dtype=bool)
    n_bursts, burst_height, burst_step = 9, 80, BURST_STEP
    subswath_width, overlap = 400, 30
    for subswath in range(3):
        col0 = 40 + subswath * (subswath_width - overlap)
        for burst in range(n_bursts):
            row0 = 20 + burst * burst_height + subswath * burst_step
            shift = burst * burst_step
            footprint[row0:row0 + burst_height, col0 + shift:col0 + shift + subswath_width] = True
    return footprint


def synthetic_scene(footprint: np.ndarray) -> np.ndarray:
    rows, cols = np.indices(SHAPE)
    rng = np.random.default_rng(seed=20230413)
    field = 2.0 + np.sin(rows / 37.0) + np.cos(cols / 53.0) + 0.3 * rng.standard_normal(SHAPE)
    field[LAND] = 50.0
    return np.where(footprint, field, 0.0)


def inference_for(footprint: np.ndarray, vv: np.ndarray) -> OPRInference:
    return OPRInference(
        vv_nice_display=vv,
        vh_nice_display=vv,
        vh_nesz_in=np.ones(SHAPE),
        ecmwf_wind_speed=np.ones(SHAPE),
        incidence=np.full(SHAPE, 35.0),
        invalidity_mask=np.zeros(SHAPE, dtype=np.uint8),
        footprint=footprint,
        models=(),
    )


@pytest.fixture(scope="module")
def footprint() -> np.ndarray:
    return staircase_footprint()


@pytest.fixture(scope="module")
def scene(footprint: np.ndarray) -> np.ndarray:
    return synthetic_scene(footprint)


@pytest.fixture(scope="module")
def inference(footprint: np.ndarray, scene: np.ndarray) -> OPRInference:
    return inference_for(footprint, scene)


@pytest.fixture(scope="module")
def padded(inference: OPRInference, scene: np.ndarray) -> np.ndarray:
    return get_inputs((scene,), inference.reflection_indices)[:, :, 0]


@pytest.fixture(scope="module")
def padded_footprint(inference: OPRInference, footprint: np.ndarray) -> np.ndarray:
    out = np.zeros(inference.padded_shape[:2], dtype=bool)
    out[inference.pad:inference.pad + SHAPE[0], inference.pad:inference.pad + SHAPE[1]] = footprint
    return out


def test_footprint_rows_and_columns_are_single_intervals(footprint: np.ndarray) -> None:
    for axis in (0, 1):
        ok, first, last = line_limits(footprint, axis)
        counts = footprint.sum(axis=axis)
        assert np.array_equal(counts[ok], (last - first + 1)[ok])


def test_inside_footprint_is_unchanged(inference, padded, padded_footprint, scene, footprint) -> None:
    pad = inference.pad
    inner = padded[pad:pad + SHAPE[0], pad:pad + SHAPE[1]]
    assert np.array_equal(inner[footprint], scene[footprint])
    assert padded.shape == inference.padded_shape[:2]


def test_no_zero_in_padded_input(padded: np.ndarray) -> None:
    assert np.all(np.isfinite(padded))
    assert np.all(padded != 0)


def test_outside_pixels_mirror_about_the_bounding_box(inference, padded_footprint) -> None:
    pad = inference.pad
    rows_src, cols_src = inference.reflection_indices
    rows, cols = np.indices(padded_footprint.shape)
    rows_ok, _, _ = line_limits(padded_footprint, 1)
    cols_ok, _, _ = line_limits(padded_footprint, 0)
    r0, r1 = np.flatnonzero(rows_ok)[[0, -1]]
    c0, c1 = np.flatnonzero(cols_ok)[[0, -1]]
    mirror_rows, mirror_cols = fold(rows, r0, r1), fold(cols, c0, c1)
    selection = ~padded_footprint & padded_footprint[mirror_rows, mirror_cols]
    corners = ~rows_ok[:, None] & ~cols_ok[None, :]
    assert (selection & corners).any()
    assert np.array_equal(rows_src[selection] + pad, mirror_rows[selection])
    assert np.array_equal(cols_src[selection] + pad, mirror_cols[selection])
    flat = np.ravel_multi_index((rows_src[corners], cols_src[corners]), SHAPE)
    assert np.unique(flat, return_counts=True)[1].max() <= 4


def test_source_map_is_locally_continuous_outside(inference, padded_footprint) -> None:
    rows_src, cols_src = inference.reflection_indices
    outside = ~padded_footprint
    jumps = []
    for axis in (0, 1):
        jump = np.maximum(np.abs(np.diff(rows_src, axis=axis)), np.abs(np.diff(cols_src, axis=axis)))
        pairs = (outside[1:, :] & outside[:-1, :]) if axis == 0 else (outside[:, 1:] & outside[:, :-1])
        jumps.append(jump[pairs])
    jumps = np.concatenate(jumps)
    assert np.percentile(jumps, 99.9) <= 2 * BURST_STEP
    assert jumps.max() <= 2 * BURST_STEP + 1
    flat = np.ravel_multi_index((rows_src[outside], cols_src[outside]), SHAPE)
    assert np.unique(flat, return_counts=True)[1].max() <= 16

def test_boundary_continuity(inference, padded, padded_footprint) -> None:
    inner_pairs = padded_footprint[:, 1:] & padded_footprint[:, :-1]
    edge_pairs = padded_footprint[:, 1:] ^ padded_footprint[:, :-1]
    land = np.zeros_like(padded_footprint)
    land[inference.pad:inference.pad + SHAPE[0], inference.pad:inference.pad + SHAPE[1]][LAND] = True
    near_land = land[:, 1:] | land[:, :-1]
    jumps = np.abs(np.diff(padded, axis=1))
    inside = jumps[inner_pairs & ~near_land].mean()
    across = jumps[edge_pairs].mean()
    assert across <= 2 * inside
    inner_pairs = padded_footprint[1:, :] & padded_footprint[:-1, :]
    edge_pairs = padded_footprint[1:, :] ^ padded_footprint[:-1, :]
    near_land = land[1:, :] | land[:-1, :]
    jumps = np.abs(np.diff(padded, axis=0))
    assert jumps[edge_pairs].mean() <= 2 * jumps[inner_pairs & ~near_land].mean()


def test_land_block_is_never_a_mirror_axis(inference, footprint, scene, padded_footprint) -> None:
    pad = inference.pad
    land_nan = scene.copy()
    land_nan[LAND] = np.nan
    out = get_inputs((land_nan,), inference.reflection_indices)[:, :, 0]
    assert not np.isnan(out[~padded_footprint]).any()
    assert np.isnan(out[pad:pad + SHAPE[0], pad:pad + SHAPE[1]][LAND]).all()


def test_padded_mask_is_false_outside_footprint(inference, padded_footprint) -> None:
    assert inference.padded_mask.shape == padded_footprint.shape
    assert not inference.padded_mask[~padded_footprint].any()
    pad = inference.pad
    assert np.array_equal(inference.padded_mask[pad:pad + SHAPE[0], pad:pad + SHAPE[1]], inference.mask)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (130, 200), (833, 1328)])
def test_full_footprint_matches_rectangle_reflection(shape: typing.Tuple[int, int]) -> None:
    rng = np.random.default_rng(seed=0)
    arrays = tuple(rng.random(shape) for _ in range(3))
    inference = OPRInference(
        vv_nice_display=arrays[0],
        vh_nice_display=arrays[1],
        vh_nesz_in=arrays[2],
        ecmwf_wind_speed=arrays[2],
        incidence=arrays[2],
        invalidity_mask=np.zeros(shape, dtype=np.uint8),
        footprint=np.ones(shape, dtype=bool),
        models=(),
    )
    pad, pad_x, pad_y = inference.pad, inference.pad_x, inference.pad_y
    indices = footprint_reflection_indices(np.ones(shape, dtype=bool), pad_x, pad_y, pad)
    expected = np.pad(np.stack(arrays, axis=-1), ((pad, pad_x), (pad, pad_y), (0, 0)), mode="reflect")
    assert np.array_equal(get_inputs(arrays, indices), expected)


class _ConstantModel:
    normalization = np.stack([np.zeros(5), np.ones(5)])

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, inputs, batch_size=None):
        n, size = inputs[0].shape[0], inputs[0].shape[1]
        return np.zeros((n, size, size, 1)), np.full((n, size, size, 1), self.value)


@pytest.mark.parametrize("values", [(2.0, 4.0), (0.0, 0.0)])
def test_mean_and_std_are_nan_outside_footprint(footprint, scene, values) -> None:
    inference = OPRInference(
        vv_nice_display=scene,
        vh_nice_display=scene,
        vh_nesz_in=np.ones(SHAPE),
        ecmwf_wind_speed=np.ones(SHAPE),
        incidence=np.full(SHAPE, 35.0),
        invalidity_mask=np.zeros(SHAPE, dtype=np.uint8),
        footprint=footprint,
        models=tuple(_ConstantModel(value) for value in values),
    )
    mean, relative_std = inference.rain_rate_ensemble()
    assert np.all(np.isnan(mean[~footprint]))
    assert np.all(np.isnan(relative_std[~footprint]))
    assert np.isfinite(mean[inference.mask]).mean() > 0.99
    assert np.isfinite(relative_std[inference.mask]).mean() > 0.99
