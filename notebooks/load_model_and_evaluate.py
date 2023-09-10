# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from sklearn.metrics import classification_report
import numpy as np

# Commented out IPython magic to ensure Python compatibility.

# import the necessary packages
base_path = os.getcwd()
# define the names of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "val"
# set the batch size when fine-tuning
cl = {"1": "Scratch", "2": "Dent", "3": "Rim", "4": "Other"}
CLASSES = ["scratch", "dent", "rim", "other"]

base_path = "./dataset/"

testPath = os.path.sep.join([base_path, TEST])

# determine the total number of image paths in training, validation,
# and testing directories
BATCH_SIZE = 32

valAug = ImageDataGenerator()
testGen = valAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE,
)


def evaluate_model(model_x, name):
    pred_idxs = model_x.predict(x=testGen)
    pred_idxs = np.argmax(pred_idxs, axis=1)
    print(f"                Classification Report of {name}")

    print(
        classification_report(
            testGen.classes,
            pred_idxs,
            target_names=testGen.class_indices.keys(),
        )
    )


model_resnet = load_model("./model/model.h5")
evaluate_model(model_resnet, "Resnet50")
