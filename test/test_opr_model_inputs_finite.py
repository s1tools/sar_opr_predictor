import json
import os
import typing
from pathlib import Path

import numpy as np
import pytest

from s1tools.opr_predictor.opr_inference import OPRInference, YesUGAN
from test_opr_footprint_reflection import SHAPE, staircase_footprint, synthetic_scene


PROJECT_ROOT: Path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
REFERENCE_NPZ: Path = PROJECT_ROOT / "test" / "ressources" / "opr" / "20230413T063223.npz"
MODEL_JSON: str = os.path.join(PROJECT_ROOT, "models.json")
INPUT_NAMES: typing.Tuple[str, ...] = ("vv_nice_display", "vh_nice_display", "vh_nesz_in", "ecmwf_wind_speed", "incidence")


@pytest.fixture(scope="module")
def model():
    with open(MODEL_JSON, "r") as file:
        data = json.load(file)
    info = next(iter(data["iw"].values()))
    gan = YesUGAN(folder=PROJECT_ROOT / data["Dir"], h5_filename=info["h5"], normalization_filename=info["normalization"])
    gan.load()
    return gan.generator


def captured_model_inputs(
    monkeypatch: pytest.MonkeyPatch,
    model,
    inference: OPRInference,
) -> typing.Tuple[typing.List[np.ndarray], np.ndarray]:
    captured: typing.List[np.ndarray] = []
    predict = model.predict

    def capturing_predict(inputs, *args, **kwargs):
        captured.extend(np.asarray(array) for array in inputs)
        return predict(inputs, *args, **kwargs)

    monkeypatch.setattr(model, "predict", capturing_predict)
    prediction = inference.run_model(model)
    return captured, prediction


def staircase_inference_with_nan(model) -> OPRInference:
    footprint = staircase_footprint()
    vv = synthetic_scene(footprint)
    inputs = {
        "vv_nice_display": vv,
        "vh_nice_display": vv.copy(),
        "vh_nesz_in": np.ones(SHAPE),
        "ecmwf_wind_speed": np.full(SHAPE, 8.0),
        "incidence": np.full(SHAPE, 35.0),
    }
    for array in inputs.values():
        array[~footprint] = np.nan
    invalidity_mask = np.zeros(SHAPE, dtype=np.uint8)
    inside_rows, inside_cols = np.nonzero(footprint)
    rng = np.random.default_rng(seed=7)
    picks = rng.choice(inside_rows.size, size=4 * len(INPUT_NAMES), replace=False).reshape(len(INPUT_NAMES), 4)
    for name, pick in zip(INPUT_NAMES, picks):
        rows, cols = inside_rows[pick], inside_cols[pick]
        inputs[name][rows, cols] = np.nan
        invalidity_mask[rows[:2], cols[:2]] = 1
    return OPRInference(invalidity_mask=invalidity_mask, footprint=footprint, models=(model,), **inputs)


def test_staircase_scene_with_nan_gives_finite_model_inputs(monkeypatch: pytest.MonkeyPatch, model) -> None:
    inference = staircase_inference_with_nan(model)
    captured, prediction = captured_model_inputs(monkeypatch, model, inference)
    assert len(captured) == 2
    for array in captured:
        assert array.size > 0
        assert np.isfinite(array).all()
    any_nan = np.logical_or.reduce([~np.isfinite(getattr(inference, name)) for name in INPUT_NAMES])
    assert np.isnan(prediction[any_nan]).all()
    assert np.isfinite(prediction[inference.mask]).all()


def test_real_scene_gives_finite_model_inputs(monkeypatch: pytest.MonkeyPatch, model) -> None:
    with open(REFERENCE_NPZ, "rb") as file:
        data = {key: value for key, value in np.load(file).items()}
    inference = OPRInference(
        vv_nice_display=data["vv_nice_display"],
        vh_nice_display=data["vh_nice_display"],
        vh_nesz_in=data["vh_nesz_in"],
        ecmwf_wind_speed=data["ecmwf_wind_speed"],
        incidence=data["incidence"],
        invalidity_mask=data["validity_mask"],
        footprint=data["footprint"],
        models=(model,),
    )
    captured, _ = captured_model_inputs(monkeypatch, model, inference)
    assert len(captured) == 2
    for array in captured:
        assert array.size > 0
        assert np.isfinite(array).all()
