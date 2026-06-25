import json
import typing
from pathlib import Path

import numpy as np
import pytest

from s1tools.opr_predictor.opr_inference import (
    CONFIG,
    OPRInference,
    YesUGAN,
    block_mean_downsample,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_NPZ = PROJECT_ROOT / "test" / "ressources" / "opr" / "20230413T063223.npz"
MODEL_JSON = PROJECT_ROOT / "models.json"
PROOF_OUTPUT_DIR = PROJECT_ROOT / "test" / ".tmp" / "opr_pipeline_proof"
PROOF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROOF_NPZ_PATH = PROOF_OUTPUT_DIR / "ipf_opr_reference.npz"


@pytest.fixture(scope="module")
def reference_inputs() -> typing.Dict[str, np.ndarray]:
    with open(REFERENCE_NPZ, "rb") as file:
        return {key: value for key, value in np.load(file).items()}

@pytest.fixture(scope="module")
def ipf_models() -> typing.Tuple:
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
def ipf_inference(reference_inputs, ipf_models):
    return OPRInference(
        vv_nice_display=reference_inputs["vv_nice_display"],
        vh_nice_display=reference_inputs["vh_nice_display"],
        vh_nesz_in=reference_inputs["vh_nesz_in"],
        ecmwf_wind_speed=reference_inputs["ecmwf_wind_speed"],
        incidence=reference_inputs["incidence"],
        invalidity_mask=reference_inputs["validity_mask"],
        models=ipf_models,
    )


def test_models_load(ipf_models) -> None:
    import tensorflow as tf

    assert len(ipf_models) == 4
    for model in ipf_models:
        assert isinstance(model, tf.keras.models.Model)
        assert hasattr(model, "normalization")
        assert model.normalization.shape == (2, 5)


def test_inference_mean_and_rstd(ipf_inference) -> None:
    mean_200m, rstd_200m = ipf_inference.rain_rate_ensemble()
    assert mean_200m.shape == (833, 1328)
    assert rstd_200m.shape == (833, 1328)
    assert np.isfinite(mean_200m).any()
    assert np.isfinite(rstd_200m).any()
    assert np.nanmin(mean_200m) >= 0
    assert np.nanmax(mean_200m) <= 60
    assert np.nanmin(rstd_200m) >= 0


def test_downsample_to_1km(ipf_inference) -> None:
    mean_200m, rstd_200m = ipf_inference.rain_rate_ensemble()
    factor = int(round(CONFIG.output_resolution_m / CONFIG.target_resolution_m))
    assert factor == 5
    mean_1km = block_mean_downsample(mean_200m, factor)
    rstd_1km = block_mean_downsample(rstd_200m, factor)
    assert mean_1km.shape == (833 // factor, 1328 // factor)
    assert rstd_1km.shape == (833 // factor, 1328 // factor)


def test_npz_emission(ipf_inference) -> None:
    mean_200m, rstd_200m = ipf_inference.rain_rate_ensemble()
    factor = int(round(CONFIG.output_resolution_m / CONFIG.target_resolution_m))
    mean_1km = block_mean_downsample(mean_200m, factor).astype(np.float32)
    rstd_1km = block_mean_downsample(rstd_200m, factor).astype(np.float32)

    np.savez(
        PROOF_NPZ_PATH,
        owiPrecipitationRate=mean_1km,
        owiPrecipitationStd=rstd_1km,
    )
    assert PROOF_NPZ_PATH.exists()

    reloaded = np.load(PROOF_NPZ_PATH)
    assert set(reloaded.files) == {
        "owiPrecipitationRate",
        "owiPrecipitationStd",
    }
    np.testing.assert_array_equal(reloaded["owiPrecipitationRate"], mean_1km)
    np.testing.assert_array_equal(reloaded["owiPrecipitationStd"], rstd_1km)
