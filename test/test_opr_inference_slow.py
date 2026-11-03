import json
import os
import typing

import numpy as np
import pytest
from pathlib import Path
from s1tools.opr_predictor.opr_inference import (
    CONFIG,
    OPRInference,
    YesUGAN,
)


PROJECT_ROOT: Path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
REFERENCE_NPZ: str = PROJECT_ROOT / "test" / "ressources" / "opr" / "20230413T063223.npz"
MODEL_JSON: str = os.path.join(PROJECT_ROOT, "models.json")
EXPECTED_SHAPE: typing.Tuple[int, int] = (833, 1328)


def assert_array_integrity(
    array: np.ndarray,
    shape: tuple[int, ...] | None = None,
    min_value: typing.Optional[float] = None,
    mean_value: typing.Optional[float] = None,
    max_value: typing.Optional[float] = None,
    decimals: int = 0,
) -> None:
    """Check shape, NaN-aware min/mean/max within ``10 ** (-decimals)``; collect all failures."""
    assert isinstance(array, np.ndarray), f"Expected ndarray, got {type(array)}"
    failures: typing.List[str] = []
    if shape is not None and shape != array.shape:
        failures.append(f"Shape mismatch: expected {shape}, got {array.shape}")
    tolerance = 10 ** (-decimals)
    if min_value is not None and not np.isclose(np.nanmin(array), min_value, rtol=0, atol=tolerance):
        failures.append(f"Min mismatch: expected {min_value}, got {np.nanmin(array)}")
    if mean_value is not None and not np.isclose(np.nanmean(array), mean_value, rtol=0, atol=tolerance):
        failures.append(f"Mean mismatch: expected {mean_value}, got {np.nanmean(array)}")
    if max_value is not None and not np.isclose(np.nanmax(array), max_value, rtol=0, atol=tolerance):
        failures.append(f"Max mismatch: expected {max_value}, got {np.nanmax(array)}")
    if failures:
        pytest.fail("\n".join(failures))


@pytest.fixture(scope="module")
def data_from_reference() -> typing.Dict[str, np.ndarray]:
    with open(REFERENCE_NPZ, "rb") as file:
        return {key: value for key, value in np.load(file).items()}


@pytest.fixture(scope="module")
def models() -> typing.Tuple:
    loaded = []
    with open(MODEL_JSON, "r" ) as file:
        data = json.load(file)
    folder = PROJECT_ROOT / data["Dir"]
    for info in data["iw"].values():
        gan = YesUGAN(folder=folder, h5_filename=info["h5"], normalization_filename=info["normalization"])
        gan.load()
        loaded.append(gan.generator)
    return tuple(loaded)


@pytest.fixture(scope="module")
def sar_data(
    data_from_reference: typing.Dict[str, np.ndarray],
    models: typing.Tuple,
) -> OPRInference:
    return OPRInference(
        vv_nice_display=data_from_reference["vv_nice_display"],
        vh_nice_display=data_from_reference["vh_nice_display"],
        vh_nesz_in=data_from_reference["vh_nesz_in"],
        ecmwf_wind_speed=data_from_reference["ecmwf_wind_speed"],
        incidence=data_from_reference["incidence"],
        invalidity_mask=data_from_reference["validity_mask"],  # naming discrepancy in test .npz
        footprint=data_from_reference["footprint"],
        models=models,
    )


@pytest.fixture(scope="module")
def normalized_batch(sar_data: OPRInference) -> typing.Tuple[np.ndarray, np.ndarray]:
    return sar_data.get_normalized_batch(sar_data.models[0].normalization)


@pytest.fixture(scope="module")
def single_model_prediction(sar_data: OPRInference) -> np.ndarray:
    """Run the real (patch-skipping) inference path for a single model."""
    return sar_data.run_model(sar_data.models[0])


def test_reference_incidence(data_from_reference: typing.Dict[str, np.ndarray]) -> None:
    assert_array_integrity(data_from_reference["incidence"], EXPECTED_SHAPE, 30.12, 38.60, 46.14, 2)


def test_reference_wind_speed(data_from_reference: typing.Dict[str, np.ndarray]) -> None:
    assert_array_integrity(data_from_reference["ecmwf_wind_speed"], EXPECTED_SHAPE, 3.22, 9.27, 13.29, 2)


def test_reference_vv_ssr(data_from_reference: typing.Dict[str, np.ndarray]) -> None:
    assert_array_integrity(data_from_reference["vv_ssr"], EXPECTED_SHAPE, 0, 16526, 65535)


def test_computed_vv_ssr(sar_data: OPRInference) -> None:
    assert_array_integrity(sar_data.vv_ssr, EXPECTED_SHAPE, 0, 16526, 65535)


def test_reference_vh_ssr(data_from_reference: typing.Dict[str, np.ndarray]) -> None:
    assert_array_integrity(data_from_reference["vh_ssr"], EXPECTED_SHAPE, 0, 6443, 65535)


def test_computed_vh_ssr(sar_data: OPRInference) -> None:
    assert_array_integrity(sar_data.vh_ssr, EXPECTED_SHAPE, 0, 6443, 65535)


def test_reference_vh_nesz(data_from_reference: typing.Dict[str, np.ndarray]) -> None:
    assert_array_integrity(data_from_reference["vh_nesz_out"], EXPECTED_SHAPE, 0, 1866, 65535)


def test_computed_vh_nesz(sar_data: OPRInference) -> None:
    assert_array_integrity(sar_data.vh_nesz_out, EXPECTED_SHAPE, 1, 3.54, 10, 2)


def test_validity_mask(data_from_reference: typing.Dict[str, np.ndarray]) -> None:
    assert_array_integrity(data_from_reference["validity_mask"], EXPECTED_SHAPE, 0, 0.13, 1, 2)


def test_batch_image_shape(sar_data: OPRInference) -> None:
    assert_array_integrity(sar_data.batch_image, (96, 256, 256, 3))


def test_batch_scalar_shape(sar_data: OPRInference) -> None:
    assert_array_integrity(sar_data.batch_scalar, (96, 3))


def test_pad(sar_data: OPRInference) -> None:
    assert sar_data.pad == 128


def test_pad_x(sar_data: OPRInference) -> None:
    assert sar_data.pad_x == 191


def test_pad_y(sar_data: OPRInference) -> None:
    assert sar_data.pad_y == 208


def test_padded_shape(sar_data: OPRInference) -> None:
    assert sar_data.padded_shape == (1152, 1664, 3)


def test_models_existence(models: typing.Tuple) -> None:
    import tensorflow as tf

    for model in models:
        assert isinstance(model, tf.keras.models.Model)


def test_normalization_existence(models: typing.Tuple) -> None:
    for model in models:
        assert isinstance(model.normalization, np.ndarray)


def test_normalized_batch_image(normalized_batch: typing.Tuple[np.ndarray, np.ndarray]) -> None:
    assert_array_integrity(normalized_batch[0], (96, 256, 256, 3), -1.28, 0.85, 15.37, 2)


def test_normalized_batch_scalar(normalized_batch: typing.Tuple[np.ndarray, np.ndarray]) -> None:
    assert_array_integrity(normalized_batch[1], (96, 3), -2.73, -0.77, 1.90, 2)


def test_run_model_invariants(sar_data: OPRInference, single_model_prediction: np.ndarray) -> None:
    """The patch-skipping inference path yields a masked rain-rate map with valid range."""
    masked = single_model_prediction
    assert masked.shape == EXPECTED_SHAPE

    finite = np.isfinite(masked)
    assert finite.any()
    assert np.nanmax(masked) > 1.0


def test_run_model_skips_invalid_patches(sar_data: OPRInference, single_model_prediction: np.ndarray) -> None:
    """Pixels outside the mask are NaN; finite pixels live inside the mask."""
    masked = single_model_prediction
    assert np.all(np.isnan(masked[~(sar_data.mask > 0.5)]))


def test_rain_rate_ensemble(sar_data: OPRInference) -> None:
    mean, relative_std = sar_data.rain_rate_ensemble()
    assert mean.shape == EXPECTED_SHAPE
    assert relative_std.shape == EXPECTED_SHAPE
    assert np.isfinite(mean).any()
    assert np.nanmin(mean) >= 0
    assert np.nanmax(mean) <= 60
    assert np.nanmin(relative_std) >= 0
