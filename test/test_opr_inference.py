import json
import os
import typing

import numpy as np
import pytest
from pathlib import Path
from s1tools.opr_predictor.opr_inference import (
    OPRInference,
    YesUGAN,
    normalize,
)


PROJECT_ROOT: Path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
REFERENCE_NPZ: str = PROJECT_ROOT / "test" / "ressources" / "opr" / "20230413T063223_patch.npz"
MODEL_JSON: str = os.path.join(PROJECT_ROOT, "models.json")


@pytest.fixture(scope="module")
def data_from_reference()-> typing.Dict[str, np.ndarray]:
    with open(REFERENCE_NPZ, "rb") as file:
        return {key: value for key, value in np.load(file).items()}


@pytest.fixture(scope="module")
def model():
    with open(MODEL_JSON, "r") as file:
        data = json.load(file)
    folder = PROJECT_ROOT / data["Dir"]
    info = next(iter(data["iw"].values()))
    gan = YesUGAN(folder=folder, h5_filename=info["h5"], normalization_filename=info["normalization"])
    gan.load()
    return gan.generator


def test_inference_prediction(model, data_from_reference: typing.Dict[str, np.ndarray])-> None:
    image_input, scalar_input = normalize(
        model.normalization,
        data_from_reference["patch_image"],
        data_from_reference["patch_scalar"],
    )
    opr_prediction = model.predict((image_input, scalar_input))[1]
    assert np.allclose(opr_prediction, data_from_reference["opr_model0"])


def test_run_model_all_patches_invalid(model)-> None:
    shape: typing.Tuple[int, int] = (512, 512)
    inference = OPRInference(
        vv_nice_display=np.ones(shape, dtype=np.float64),
        vh_nice_display=np.ones(shape, dtype=np.float64),
        vh_nesz_in=np.zeros(shape, dtype=np.float64),
        ecmwf_wind_speed=np.zeros(shape, dtype=np.float64),
        incidence=np.zeros(shape, dtype=np.float64),
        invalidity_mask=np.ones(shape, dtype=np.uint8),
        footprint=np.ones(shape, dtype=bool),
        models=(model,),
    )
    assert not inference.mask.any()
    prediction = inference.run_model(model)
    assert prediction.shape == shape
    assert np.isnan(prediction).all()


def test_relative_std_zero_where_all_models_predict_zero() -> None:
    outputs = [np.array([[0.0, 1.0, np.nan]]) for _ in range(4)]
    inference = OPRInference.__new__(OPRInference)
    inference.models = list(range(4))
    inference.run_model = lambda model, batch_size=None: outputs[model]
    mean, relative_std = inference.rain_rate_ensemble()
    assert relative_std[0, 0] == 0.0
    assert relative_std[0, 1] == 0.0
    assert np.isnan(mean[0, 2]) and np.isnan(relative_std[0, 2])
