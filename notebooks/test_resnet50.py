# -*- coding: utf-8 -*-

# import pytest
# import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
import os
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_resnet50():
    savedmodel = load_model("./model/model.h5")
    return savedmodel


def test_resnet50():
    savedmodel = load_resnet50()

    base_path = "./dataset/"
    test = "test"

    batch_size = 32

    testpath = os.path.sep.join([base_path, test])

    # normalize test images
    valaug = ImageDataGenerator()
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    valaug.mean = mean
    # load test images
    testgen = valaug.flow_from_directory(
        testpath,
        class_mode="categorical",
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    predidxs = savedmodel.predict(x=testgen)
    predidxs = np.argmax(predidxs, axis=1)

    y_true = testgen.classes
    y_pred = predidxs
    test_accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy of model on testset = %.2f  " % test_accuracy)

    assert test_accuracy >= 0.75


test_resnet50()
