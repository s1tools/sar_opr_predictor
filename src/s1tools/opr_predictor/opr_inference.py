import json
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


for _gpu_device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu_device, True)


DEFAULT_MODEL_KEYS: typing.Tuple[str, ...] = (
    "20230629T162527",
    "20230918T040436",
    "20231021T013609",
    "20231021T100537",
)


class LazyProperty:
    def __init__(self, func: typing.Callable) -> None:
        self.func = func
        self.attr_name = f"_lazy_{func.__name__}"

    def __get__(self, instance: typing.Any, owner: typing.Any) -> typing.Any:
        if instance is None:
            return self
        value = instance.__dict__.get(self.attr_name)
        if value is None:
            value = self.func(instance)
            instance.__dict__[self.attr_name] = value
        return value


@dataclass(frozen=True)
class OPRConfig:
    patch_size: int = 256
    batch_size: int = 16
    input_shape: typing.Tuple[int, int, int] = (256, 256, 3)
    layer_kernels: typing.Tuple[int, ...] = (32, 64, 128, 256)
    n_scalar_inputs: int = 3
    vv_clip: float = 6.0
    vh_clip: float = 12.0
    nesz_clip: float = 50.0
    target_resolution_m: int = 200
    output_resolution_m: int = 1000
    minimum_incidence_angle: float = 29.1

    regression_floor_mm_per_hour: float = 0.5
    h5_filename: str = "G.h5"
    normalization_filename: str = "normalization.npy"
    model_keys: typing.Tuple[str, ...] = DEFAULT_MODEL_KEYS


CONFIG: OPRConfig = OPRConfig()


def padding_layer(layer: tf.Tensor) -> tf.Tensor:
    return tf.keras.layers.Lambda(
        lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    )(layer)


def convolution_block(layer: tf.Tensor, n_kernels: int) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    for _ in range(2):
        layer = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation="relu", padding="valid")(layer)
        layer = padding_layer(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    pooled_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
    return pooled_layer, layer


def deconvolution_block(current_layer: tf.Tensor, inherited_layer: tf.Tensor, n_kernels: int) -> tf.Tensor:
    current_layer = tf.keras.layers.UpSampling2D(size=(2, 2))(current_layer)
    current_layer = tf.keras.layers.Conv2D(n_kernels, 2, activation="relu", padding="same")(current_layer)
    current_layer = tf.keras.layers.concatenate([current_layer, inherited_layer], axis=3)
    for _ in range(2):
        current_layer = tf.keras.layers.Conv2D(n_kernels, (3, 3), activation="relu", padding="valid")(current_layer)
        current_layer = padding_layer(current_layer)
    current_layer = tf.keras.layers.BatchNormalization()(current_layer)
    return current_layer


def build_encoder(
    current_layer: tf.Tensor,
    layer_kernels: typing.Tuple[int, ...],
) -> typing.Tuple[tf.Tensor, typing.Tuple[tf.Tensor, ...]]:
    inherited_layers = []
    for n_kernels in layer_kernels:
        current_layer, unpooled_layer = convolution_block(current_layer, n_kernels)
        inherited_layers.append(unpooled_layer)
    return current_layer, tuple(inherited_layers)


def build_decoder(
    current_layer: tf.Tensor,
    inherited_layers: typing.Tuple[tf.Tensor, ...],
    layer_kernels: typing.Tuple[int, ...],
) -> tf.Tensor:
    for n_kernels, inherited_layer in zip(layer_kernels[::-1], inherited_layers[::-1]):
        current_layer = deconvolution_block(current_layer, inherited_layer, n_kernels)
    return current_layer


def expand_inputs(x: tf.Tensor) -> tf.Tensor:
    return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)


@dataclass
class YesUGAN:
    folder: str
    h5_filename: str = CONFIG.h5_filename
    normalization_filename: str = CONFIG.normalization_filename
    input_shape: typing.Tuple[int, int, int] = CONFIG.input_shape
    layer_kernels: typing.Tuple[int, ...] = CONFIG.layer_kernels
    n_scalar_inputs: int = CONFIG.n_scalar_inputs

    def __post_init__(self) -> None:
        K.clear_session()

    @LazyProperty
    def generator(self) -> tf.keras.models.Model:
        array_input_layer = tf.keras.layers.Input(self.input_shape, name="img_input")
        scalar_input_layer = tf.keras.layers.Input((self.n_scalar_inputs,), name="scalar_input")
        scalar_layer = tf.keras.layers.Lambda(expand_inputs)(scalar_input_layer)

        current_layer, inherited_layers = build_encoder(array_input_layer, self.layer_kernels)

        scalar_layer = tf.keras.layers.UpSampling2D(current_layer.shape[1:3])(scalar_layer)
        current_layer = tf.keras.layers.concatenate([current_layer, scalar_layer], axis=-1)
        current_layer = convolution_block(current_layer, 2 * self.layer_kernels[-1])[1]

        current_layer = build_decoder(current_layer, inherited_layers, self.layer_kernels)

        mask_output = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="mask_output")(current_layer)
        regression_output = tf.keras.layers.Conv2D(
            1, (1, 1), activation="relu", name="regression_output",
        )(current_layer)

        return tf.keras.models.Model(
            inputs=[array_input_layer, scalar_input_layer],
            outputs=[mask_output, regression_output],
            name="generator",
        )

    def load(self) -> None:
        self.generator.load_weights(os.path.join(self.folder, self.h5_filename), by_name=True)
        self.generator.normalization = np.load(os.path.join(self.folder, self.normalization_filename))


def get_model(folder: str) -> tf.keras.models.Model:
    gan = YesUGAN(folder=folder)
    gan.load()
    return gan.generator


def get_models(folders: typing.Tuple[str, ...]) -> typing.Tuple[tf.keras.models.Model, ...]:
    return tuple(get_model(folder) for folder in folders)

def get_models_from_aux_ml2(aux_ml2_filename, mode="IW") -> typing.Tuple[tf.keras.models.Model, ...]:
    data_path =  Path(aux_ml2_filename) / "data"
    json_file = list(data_path.glob("*.json"))[0]
    with open(json_file, "r") as f:
        jdata = json.load(f)
    model_path = data_path / jdata["Models"]["OceanPrecipitationRate"]["Dir"]
    gans = []
    for k, info in jdata["Models"]["OceanPrecipitationRate"][mode.lower()].items():
        gan = YesUGAN(folder=model_path,
                      h5_filename=info["h5"],
                      normalization_filename=info["normalization"]
                      )
        gan.load()
        gans.append(gan)
    return tuple(g.generator for g in gans)

def get_slice_generator(
    shape: typing.Tuple[int, ...],
    patch_size: int = CONFIG.patch_size,
) -> typing.Generator[typing.Tuple[slice, slice], None, None]:
    stride = patch_size // 4
    for x1 in range(0, shape[0] - patch_size, stride):
        x2 = x1 + patch_size
        for y1 in range(0, shape[1] - patch_size, stride):
            y2 = y1 + patch_size
            yield np.s_[x1:x2, y1:y2]


def get_inputs(
    arrays: typing.Tuple[np.ndarray, ...],
    pad_x: int,
    pad_y: int,
    pad: int = 0,
) -> np.ndarray:
    inputs: np.ndarray = np.stack(arrays, axis=-1)
    pad_width = ((pad, pad_x), (pad, pad_y), (0, 0))
    return np.pad(inputs, pad_width, mode="reflect")


def concatenation_to_patches(
    image_inputs: np.ndarray,
    scalar_inputs: np.ndarray,
    patch_size: int = CONFIG.patch_size,
) -> typing.Tuple[np.ndarray, np.ndarray, typing.Tuple[typing.Tuple[slice, slice], ...]]:
    batch_image: typing.List[np.ndarray] = []
    batch_scalar: typing.List[np.ndarray] = []
    coordinates: typing.List[typing.Tuple[slice, slice]] = []
    for slices in get_slice_generator(image_inputs.shape, patch_size):
        batch_image.append(image_inputs[slices])
        batch_scalar.append(np.mean(scalar_inputs[slices], axis=(0, 1)))
        coordinates.append(slices)
    return np.array(batch_image), np.array(batch_scalar), tuple(coordinates)


def content_to_patches(
    copol_ssr: np.ndarray,
    crosspol_ssr: np.ndarray,
    mask: np.ndarray,
    crosspol_nesz: np.ndarray,
    incidence_angle: np.ndarray,
    ecmwf_wind_speed: np.ndarray,
    pad_x: int,
    pad_y: int,
    pad: int = 0,
    patch_size: int = CONFIG.patch_size,
) -> typing.Tuple[np.ndarray, np.ndarray, typing.Tuple[typing.Tuple[slice, slice], ...], typing.Tuple[int, ...]]:
    image_inputs = get_inputs((copol_ssr, crosspol_ssr, mask * 0), pad_x, pad_y, pad)
    scalar_inputs = get_inputs((crosspol_nesz, incidence_angle, ecmwf_wind_speed), pad_x, pad_y, pad)
    batch_image, batch_scalar, coordinates = concatenation_to_patches(
        image_inputs, scalar_inputs, patch_size=patch_size,
    )
    return batch_image, batch_scalar, coordinates, image_inputs.shape


def unpack_mosaic_prediction(
    mosaics: typing.Tuple[np.ndarray, ...],
    shape: typing.Tuple[int, ...],
    coordinates: typing.Sequence[typing.Tuple[slice, slice]],
    pad_x: int,
    pad_y: int,
    pad: int = 0,
    patch_size: int = CONFIG.patch_size,
) -> typing.Tuple[np.ndarray, ...]:
    predictions: typing.List[np.ndarray] = []
    center_slice = np.s_[patch_size // 4:-patch_size // 4, patch_size // 4:-patch_size // 4]
    for mosaic in mosaics[:2]:
        prediction = np.full(shape[:2], np.nan)
        for (x_slice, y_slice), tile in zip(coordinates, mosaic):
            prediction[x_slice, y_slice][center_slice] = tile[center_slice][:, :, 0]
        prediction = prediction[pad:-pad_x, pad:-pad_y]
        predictions.append(prediction)
    return tuple(predictions)


def normalize(
    normalization: np.ndarray,
    image_input: np.ndarray,
    scalar_input: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray]:
    image_mean = normalization[0][:2]
    image_standard_deviation = normalization[1][:2]
    scalar_mean = normalization[0][2:]
    scalar_standard_deviation = normalization[1][2:]
    image_input = image_input.copy()
    image_input[:, :, :, :2] = (image_input[:, :, :, :2] - image_mean) / image_standard_deviation
    scalar_input = (scalar_input - scalar_mean) / scalar_standard_deviation
    return image_input, scalar_input


@dataclass
class OPRInference:
    vv_nice_display: np.ndarray
    vh_nice_display: np.ndarray
    vh_nesz_in: np.ndarray
    ecmwf_wind_speed: np.ndarray
    incidence: np.ndarray
    invalidity_mask: np.ndarray
    models: typing.Tuple[tf.keras.models.Model, ...]

    patch_size: int = CONFIG.patch_size

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return self.vv_nice_display.shape

    @LazyProperty
    def vv_ssr(self) -> np.ndarray:
        clip_value = CONFIG.vv_clip
        return np.round(np.minimum(self.vv_nice_display ** 2, clip_value)) * (2 ** 16 - 1) / clip_value

    @LazyProperty
    def vh_ssr(self) -> np.ndarray:
        clip_value = CONFIG.vh_clip
        return np.round(np.minimum(self.vh_nice_display ** 2, clip_value)) * (2 ** 16 - 1) / clip_value

    @LazyProperty
    def vh_nesz_out(self) -> np.ndarray:
        clip_value = CONFIG.nesz_clip
        return np.round(np.minimum(self.vh_nice_display ** 2, clip_value)) * (2 ** 16 - 1) / clip_value

    @LazyProperty
    def mask(self) -> np.ndarray:
        return np.logical_and.reduce((self.vv_ssr > 0, self.invalidity_mask == 0))

    @LazyProperty
    def pad(self) -> int:
        return self.patch_size // 4

    @LazyProperty
    def pad_x(self) -> int:
        pad_x = self.patch_size - (self.shape[0] + self.pad) % self.patch_size
        if pad_x < self.pad:
            pad_x += self.patch_size
        return pad_x

    @LazyProperty
    def pad_y(self) -> int:
        pad_y = self.patch_size - (self.shape[1] + self.pad) % self.patch_size
        if pad_y < self.pad:
            pad_y += self.patch_size
        return pad_y

    def set_patches(self) -> None:
        batch_image, batch_scalar, coordinates, padded_shape = content_to_patches(
            self.vv_ssr, self.vh_ssr, self.mask * 0,
            self.vh_nesz_out, self.incidence, self.ecmwf_wind_speed,
            self.pad_x, self.pad_y, self.pad, patch_size=self.patch_size,
        )
        self.__dict__["_lazy_batch_image"] = batch_image
        self.__dict__["_lazy_batch_scalar"] = batch_scalar
        self.__dict__["_lazy_coordinates"] = coordinates
        self.__dict__["_lazy_padded_shape"] = padded_shape

    @LazyProperty
    def batch_image(self) -> np.ndarray:
        self.set_patches()
        return self.__dict__["_lazy_batch_image"]

    @LazyProperty
    def batch_scalar(self) -> np.ndarray:
        self.set_patches()
        return self.__dict__["_lazy_batch_scalar"]

    @LazyProperty
    def coordinates(self) -> typing.Tuple[typing.Tuple[slice, slice], ...]:
        self.set_patches()
        return self.__dict__["_lazy_coordinates"]

    @LazyProperty
    def padded_shape(self) -> typing.Tuple[int, ...]:
        self.set_patches()
        return self.__dict__["_lazy_padded_shape"]

    def get_normalized_batch(self, normalization: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return normalize(normalization, self.batch_image, self.batch_scalar)

    def unpack_mosaic_prediction(self, mosaics: typing.Tuple[np.ndarray, ...]) -> typing.Tuple[np.ndarray, ...]:
        return unpack_mosaic_prediction(
            mosaics, self.padded_shape, self.coordinates,
            self.pad_x, self.pad_y, pad=self.pad, patch_size=self.patch_size,
        )

    def run_model(
            self, 
            model: tf.keras.models.Model, 
            batch_size: int = CONFIG.patch_size
        ) -> np.ndarray:
        image_input, scalar_input = self.get_normalized_batch(model.normalization)

        are_patches_valid: typing.List[bool] = [
            self.mask[coordinate].mean() > 0.5
            for coordinate in self.coordinates
        ]
        valid_indices = np.flatnonzero(are_patches_valid)
        n_patches = len(self.coordinates)

        outputs = model.predict(
            (image_input[valid_indices], scalar_input[valid_indices]),
            batch_size=batch_size,
        )

        mosaics = []
        for output in outputs:
            full = np.zeros((n_patches, *output.shape[1:]), dtype=output.dtype)
            full[valid_indices] = output
            mosaics.append(full)

        predictions = self.unpack_mosaic_prediction(mosaics)
        regression = predictions[1]
        thresholded = np.where(regression >= CONFIG.regression_floor_mm_per_hour, regression, 0.0)
        return np.where(self.mask > 0.5, thresholded, np.nan)

    def rain_rate_ensemble(self, batch_size: int = CONFIG.patch_size) -> typing.Tuple[np.ndarray, np.ndarray]:
        stacked = np.stack([self.run_model(model, batch_size) for model in self.models], axis=0)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_std = np.where(mean > 0, std / mean, np.nan)
        return mean, relative_std


def block_mean_downsample(array: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return array
    ny, nx = array.shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor
    trimmed = array[:ny_trim, :nx_trim]
    reshaped = trimmed.reshape(ny_trim // factor, factor, nx_trim // factor, factor)
    return np.nanmean(np.nanmean(reshaped, axis=3), axis=1)
