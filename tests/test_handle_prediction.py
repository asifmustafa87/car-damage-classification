from unittest.mock import create_autospec

import imgaug.augmenters
import numpy as np
import pytest
import keras

from src.ml import handle_prediction


@pytest.fixture
def load_model_mock(monkeypatch):
    model_mock_value = "model_test"
    model_spec = create_autospec(
        keras.engine.functional.Functional, return_value=model_mock_value
    )
    model_spec.layers[0].input_shape = [(None, 224, 224, 3)]
    model_spec.layers[-1].output_shape = (None, 4)

    load_model_spec = create_autospec(
        keras.models.load_model, return_value=model_spec
    )

    monkeypatch.setattr(keras.models, "load_model", load_model_spec)


@pytest.fixture
def run_single_prediction_on_model(monkeypatch):
    prediction = np.zeros((1, 4))
    model_mock = create_autospec(
        keras.engine.functional.Functional.predict,
        return_value=prediction,
    )
    monkeypatch.setattr(
        keras.engine.functional.Functional, "predict", model_mock
    )


@pytest.fixture
def image_augmentation_mock(monkeypatch):
    augmented_img = np.zeros((1, 224, 224, 3))
    imgaug_spec = create_autospec(
        imgaug.augmenters.SomeOf.__call__, return_value=augmented_img
    )

    monkeypatch.setattr(imgaug.augmenters.SomeOf, "__call__", imgaug_spec)


def test_predict(run_single_prediction_on_model, load_model_mock):
    mock_image = np.zeros((23, 578, 3))
    prediction = handle_prediction.predict(mock_image)
    assert prediction.shape == (4,)


def test_only_one_instance_of_production_model(load_model_mock):
    """Asserts that only one instance of the model is present at a given time"""

    p = handle_prediction.ProductionModels("fakepath")
    q = handle_prediction.ProductionModels("another/fake/path")
    assert p.tensorflow_model is q.tensorflow_model


def test_get_model_input_shape(load_model_mock):
    assert handle_prediction.ProductionModels(
        "fakepath"
    ).get_model_input_shape() == (1, 224, 224, 3)


def test_get_model_output_shape(load_model_mock):
    assert handle_prediction.ProductionModels(
        "fakepath"
    ).get_model_output_shape() == (1, 4)


def test_preprocess_image():
    img = np.zeros(shape=(85, 484, 3))
    new_img = handle_prediction.PreProcessedImage(img)
    assert (
        new_img.pre_processed_img.shape == (1, 224, 224, 3)
        and new_img.pre_processed_img.dtype.name == "float32"
    )


def test_un_preprocess_image():
    img = np.zeros(shape=(85, 484, 3))
    new_img = handle_prediction.PreProcessedImage(img)
    un_preprocessed_img = new_img.un_preprocess_image()
    assert un_preprocessed_img.dtype.name == "int32"


def test_augment_images(image_augmentation_mock):
    img = np.zeros(shape=(85, 484, 3))
    new_img = handle_prediction.PreProcessedImage(img)
    no_of_augment_per_img = 5
    augment_images = new_img.augment_images(no_of_augment_per_img)

    assert augment_images.shape == (no_of_augment_per_img, 224, 224, 3)
