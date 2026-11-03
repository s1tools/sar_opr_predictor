import numpy as np
import pytest

from s1tools.opr_predictor.opr_inference import CONFIG, OPRInference


def test_patch_validity_uses_patch_coordinates() -> None:
    shape = (512, 512)
    invalidity_mask = np.ones(shape, dtype=np.uint8)
    invalidity_mask[:256, :256] = 0
    inference = OPRInference(
        vv_nice_display=np.ones(shape),
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=invalidity_mask,
        footprint=np.ones(shape, dtype=bool),
        models=(),
    )
    pad = inference.pad
    assert inference.padded_mask.shape == inference.padded_shape[:2]
    n_checked = 0
    for x_slice, y_slice in inference.coordinates:
        if x_slice.start < pad or y_slice.start < pad:
            continue
        if x_slice.stop > pad + shape[0] or y_slice.stop > pad + shape[1]:
            continue
        unpadded_window = inference.mask[
            x_slice.start - pad:x_slice.stop - pad,
            y_slice.start - pad:y_slice.stop - pad,
        ]
        padded_window = inference.padded_mask[x_slice, y_slice]
        assert np.array_equal(padded_window, unpadded_window)
        n_checked += 1
    assert n_checked > 0


class _ConstantModel:
    normalization = np.stack([np.zeros(5), np.ones(5)])

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, inputs, batch_size=None):
        n = inputs[0].shape[0]
        size = inputs[0].shape[1]
        return (np.zeros((n, size, size, 1)), np.full((n, size, size, 1), self.value))


def test_rejected_tile_is_nan_where_uncovered_and_never_overwrites() -> None:
    shape = (512, 512)
    invalidity_mask = np.zeros(shape, dtype=np.uint8)
    invalidity_mask[:, :] = 1
    invalidity_mask[:, :200] = 0
    invalidity_mask[:, 400:420] = 0
    inference = OPRInference(
        vv_nice_display=np.ones(shape),
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=invalidity_mask,
        footprint=np.ones(shape, dtype=bool),
        models=(),
    )
    result = inference.run_model(_ConstantModel(5.0))
    valid = [
        inference.padded_mask[c].mean() > CONFIG.minimum_valid_tile_fraction for c in inference.coordinates
    ]
    assert not all(valid) and any(valid)
    pad = inference.pad
    covered = np.zeros(inference.padded_shape[:2], dtype=bool)
    for c, v in zip(inference.coordinates, valid):
        if v:
            covered[c] = True
    covered = covered[pad:pad + shape[0], pad:pad + shape[1]]
    expected_valid = covered & (inference.mask > 0.5)
    assert np.all(result[expected_valid] == 5.0)
    assert np.all(np.isnan(result[~expected_valid]))
    assert (~covered & (inference.mask > 0.5)).any()


def _kept_tile_count(valid_fraction: float) -> int:
    shape = (CONFIG.patch_size, CONFIG.patch_size)
    rng = np.random.default_rng(seed=0)
    invalidity_mask = np.ones(shape, dtype=np.uint8)
    n_valid = int(round(valid_fraction * invalidity_mask.size))
    invalidity_mask.flat[rng.permutation(invalidity_mask.size)[:n_valid]] = 0
    inference = OPRInference(
        vv_nice_display=np.ones(shape),
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=invalidity_mask,
        footprint=np.ones(shape, dtype=bool),
        models=(),
    )
    pad = inference.pad
    fractions = {
        (c[0].start, c[1].start): inference.padded_mask[c].mean() for c in inference.coordinates
    }
    scene_tile = fractions.pop((pad, pad))
    assert scene_tile == pytest.approx(valid_fraction, abs=0.01)
    assert all(fraction < scene_tile for fraction in fractions.values())
    result = inference.run_model(_ConstantModel(5.0))
    return int(np.isfinite(result).any())


def test_tile_with_24_percent_valid_pixels_is_rejected() -> None:
    assert _kept_tile_count(0.24) == 0


def test_tile_with_26_percent_valid_pixels_is_kept() -> None:
    assert _kept_tile_count(0.26) == 1


@pytest.mark.parametrize("size", [1, 127, 128, 129, 255, 257, 1001])
def test_every_scene_pixel_is_covered_by_four_tiles(size: int) -> None:
    shape = (size, size)
    inference = OPRInference(
        vv_nice_display=np.ones(shape),
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=np.zeros(shape, dtype=np.uint8),
        footprint=np.ones(shape, dtype=bool),
        models=(),
    )
    pad = inference.pad
    count = np.zeros(inference.padded_shape[:2], dtype=int)
    for c in inference.coordinates:
        count[c] += 1
    assert (count[pad:pad + size, pad:pad + size] == 4).all()


def test_zero_vv_inside_footprint_is_valid_data() -> None:
    shape = (CONFIG.patch_size, CONFIG.patch_size)
    vv = np.zeros(shape)
    inference = OPRInference(
        vv_nice_display=vv,
        vh_nice_display=np.ones(shape),
        vh_nesz_in=np.zeros(shape),
        ecmwf_wind_speed=np.zeros(shape),
        incidence=np.zeros(shape),
        invalidity_mask=np.zeros(shape, dtype=np.uint8),
        footprint=np.ones(shape, dtype=bool),
        models=(),
    )
    assert inference.mask.all()
    pad = inference.pad
    scene_tile = next(c for c in inference.coordinates if (c[0].start, c[1].start) == (pad, pad))
    assert inference.padded_mask[scene_tile].mean() > CONFIG.minimum_valid_tile_fraction
    result = inference.run_model(_ConstantModel(5.0))
    assert np.all(result == 5.0)
