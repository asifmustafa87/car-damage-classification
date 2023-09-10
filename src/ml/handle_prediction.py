import io
import logging
import sys
from pathlib import Path
from typing import Union

import cv2
import keras
import numpy as np
import PIL
from imgaug import augmenters as iaa
from keras.engine.functional import Functional
from numpy import uint8

from src.ml.helpers import StreamToLogger

logger = logging.getLogger(__name__)
sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

inverse_classes = {
    0: "Dent",
    1: "Other",
    2: "Rim",
    3: "Scratch",
}


class ProductionModels:
    """Class that generates a singleton object, initializes and stores model parameters"""

    _instances = {}

    def __new__(cls, path_to_model: Path, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__new__(cls, *args, **kwargs)
            logger.info("Creating new model instance %s", cls._instances[cls])
        else:
            logger.info("Returning model instance %s", cls._instances[cls])
        return cls._instances[cls]

    def __init__(
        self,
        path_to_model: Path,
    ):
        logger.info("Loading model at path %s", path_to_model)
        self.tensorflow_model: Functional = keras.models.load_model(
            path_to_model
        )

    def get_model_input_shape(self):
        """Get the input shape of the tensorflow model"""

        return (
            1,
            *self.tensorflow_model.layers[0].input_shape[0][1:],
        )

    def get_model_output_shape(self):
        """Get the output shape of the tensorflow model"""

        return (
            1,
            *self.tensorflow_model.layers[-1].output_shape[1:],
        )


class PreProcessedImage:
    """Preprocessed img"""

    img: np.ndarray

    @staticmethod
    def preprocess_image(img: Union[np.ndarray, PIL.Image.Image, bytes]):
        """Make the image ready for training with the model"""
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
        elif isinstance(img, bytes):
            img = np.array(PIL.Image.open(io.BytesIO(img)).convert("RGB"))

        img = cv2.resize(img, (224, 224))
        img = img.astype("float32")
        img = np.expand_dims(img, axis=0)
        return img

    def __init__(self, img: Union[np.ndarray, PIL.Image.Image, bytes]):
        self.pre_processed_img = self.preprocess_image(img)

    def un_preprocess_image(self):
        """Make the image ready for plotting with matplotlib"""
        unpreprocessed_image = self.pre_processed_img.astype("int32")
        return unpreprocessed_image

    def augment_images(self, augmented_imgs_per_img=3):
        """Augment cropped images"""
        augs = iaa.SomeOf(
            2,
            [
                iaa.Rotate((-20, 20)),
                iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=True),
                iaa.Dropout(p=(0, 0.1), per_channel=0.5),
                iaa.MultiplyBrightness((0.5, 1.8)),
                iaa.LogContrast(gain=(0.6, 1.4)),
                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5)),
                iaa.pillike.FilterEdgeEnhance(),
            ],
        )

        images = np.array(
            [
                PreProcessedImage(
                    augs(images=self.pre_processed_img.astype(uint8))[0]
                ).pre_processed_img.squeeze()
                for _ in range(augmented_imgs_per_img)
            ]
        )
        return images


prod_models = ProductionModels(
    path_to_model=(Path(__file__).parent.parent.parent / "model" / "model.h5"),
)


def predict(image: Union[PreProcessedImage, np.ndarray]) -> PreProcessedImage:
    if not isinstance(image, PreProcessedImage):
        img = PreProcessedImage(image)
    else:
        img = image

    logger.info("Running prediction on image with %s.predict", __name__)
    prediction = np.squeeze(
        prod_models.tensorflow_model.predict(img.pre_processed_img)
    )
    return prediction


# def predict_with_confidence(image: Union[PreProcessedImage, np.ndarray], n_iter=30, boxplot=False):
#     kdp = KerasDropoutPrediction(prod_models.tensorflow_model)
#     pred = kdp.predict(image.pre_processed_img, n_iter=n_iter)
#     pred_mean = pred.mean(axis=1)
#     pred_std = pred.std(axis=1)
#
#     # if boxplot:


def predict_with_explain(image: Union[PreProcessedImage, np.ndarray]):
    logger.info(
        "Running prediction with explainability on image on image with %s.predict_with_explain",
        __name__,
    )
    prediction = predict(image)
    predicted_label = np.argmax(prediction)
    assert isinstance(predicted_label, np.int64)

    # Disabled due to overhead.
    # explainer = tf_explain.core.GradCAM()
    # heatmap_grid = explainer.explain(
    #     (image.un_preprocess_image(), 1),
    #     prod_models.tensorflow_model,
    #     predicted_label,
    # )
    heatmap_grid = 0

    return predicted_label, prediction, heatmap_grid


def correct_model_on_images(
    image: Union[PreProcessedImage, np.ndarray],
    correction_label: int,
    augment_images: bool = False,
    augmented_imgs_per_img: int = 5,
):
    logger.info(
        "Running model retraining, correction label with %s.correct_model_on_images: %s, augment images? %s",
        __name__,
        correction_label,
        augment_images,
    )
    if not isinstance(image, PreProcessedImage):
        img = PreProcessedImage(image)
    else:
        img = image

    # Augment images or convert image to list
    if augment_images:
        logger.info(
            "Augmenting images while retraining, number of augmentation: %s",
            augmented_imgs_per_img,
        )
        logger.info("Running prediction on image")
        processed_image = img.augment_images(
            augmented_imgs_per_img=augmented_imgs_per_img
        )

    else:
        processed_image = np.array(img.pre_processed_img)
        augmented_imgs_per_img = 1

    # Preparing the label
    correction_label_ohe = np.zeros(
        (
            augmented_imgs_per_img,
            prod_models.get_model_output_shape()[-1],
        )
    )
    correction_label_ohe[:, correction_label] = 1

    model = prod_models.tensorflow_model

    model.fit(processed_image, correction_label_ohe)
