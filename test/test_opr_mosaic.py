import numpy as np

from s1tools.opr_predictor.opr_inference import (
    OPRInference,
    get_slice_generator,
    tile_weights,
    weighted_mosaic,
)


PATCH_SIZE = 256


def test_tile_weights_profile() -> None:
    weights = tile_weights(PATCH_SIZE)
    profile = 1 - np.abs(np.arange(PATCH_SIZE) - 127.5) / 128
    assert np.allclose(weights, np.outer(profile, profile))


def test_constant_field_is_rebuilt_everywhere() -> None:
    shape = (640, 896, 3)
    coordinates = tuple(get_slice_generator(shape, PATCH_SIZE))
    tiles = np.full((len(coordinates), PATCH_SIZE, PATCH_SIZE), 3.5)
    field = weighted_mosaic(tiles, coordinates, shape, PATCH_SIZE)
    assert np.allclose(field, 3.5)


def test_constant_field_includes_last_row_and_column() -> None:
    inference = OPRInference(
        vv_nice_display=np.ones((300, 410)),
        vh_nice_display=np.ones((300, 410)),
        vh_nesz_in=np.zeros((300, 410)),
        ecmwf_wind_speed=np.zeros((300, 410)),
        incidence=np.zeros((300, 410)),
        invalidity_mask=np.zeros((300, 410)),
        footprint=np.ones((300, 410), dtype=bool),
        models=(),
    )
    tiles = np.full((len(inference.coordinates), PATCH_SIZE, PATCH_SIZE, 1), 2.0)
    field, _ = inference.unpack_mosaic_prediction((tiles, tiles), inference.coordinates)
    assert field.shape == (300, 410)
    assert np.allclose(field, 2.0)
    assert np.allclose(field[-1, :], 2.0)
    assert np.allclose(field[:, -1], 2.0)


def test_single_evaluated_tile() -> None:
    shape = (512, 512, 3)
    coordinates = (np.s_[128:384, 128:384],)
    tiles = np.full((1, PATCH_SIZE, PATCH_SIZE), 7.0)
    field = weighted_mosaic(tiles, coordinates, shape, PATCH_SIZE)
    inside = np.zeros(shape[:2], dtype=bool)
    inside[128:384, 128:384] = True
    assert np.allclose(field[inside], 7.0)
    assert np.all(np.isnan(field[~inside]))
