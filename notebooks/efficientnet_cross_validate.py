# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score

import matplotlib

# matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# import the necessary packages

BASE_PATH = "./dataset/"

# define the names of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "val"
# set the batch size when fine-tuning

cl = {"1": "Scratch", "2": "Dent", "3": "Rim", "4": "Other"}
CLASSES = ["scratch", "dent", "rim", "other"]

# Train
# derive the paths to the training, validation, and testing
# directories
trainPath = os.path.sep.join([BASE_PATH, TRAIN])
valPath = os.path.sep.join([BASE_PATH, VAL])
testPath = os.path.sep.join([BASE_PATH, TEST])

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))
print(totalTrain, totalVal, totalTest)

trainAug = ImageDataGenerator()
valAug = ImageDataGenerator()

# initialize the training generator
BATCH_SIZE = 32
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE,
)
# initialize the validation generator
valGen = valAug.flow_from_directory(
    valPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE,
)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE,
)


def create_model():
    # baseModel = Xception(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
    baseModel = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)),
    )
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    # headModel = layers.GlobalAveragePooling2D(name="avg_pool")(headModel)
    # headModel = layers.BatchNormalization()(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(CLASSES), activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers[:-1]:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = SGD(learning_rate=1e-4, momentum=0.9)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return model


def cross_validate_efficientnet():
    NO_OF_EPOCHS = 10
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=1e-2,
            patience=5,
            verbose=2,
            restore_best_weights=True,
        )
    ]
    trainPath = os.path.sep.join([BASE_PATH, TRAIN])
    valPath = os.path.sep.join([BASE_PATH, VAL])
    testPath = os.path.sep.join([BASE_PATH, TEST])

    # determine the total number of image paths in training, validation,
    # and testing directories
    totalTrain = len(list(paths.list_images(trainPath)))
    totalVal = len(list(paths.list_images(valPath)))
    totalTest = len(list(paths.list_images(testPath)))
    print(totalTrain, totalVal, totalTest)
    trainAug = ImageDataGenerator()
    trainGen = trainAug.flow_from_directory(
        trainPath,
        class_mode="categorical",
        target_size=(224, 224),
        color_mode="rgb",
        shuffle=True,
        batch_size=totalTrain,
    )

    data = trainGen
    n = data.n
    print(n)
    x, y = data.next()
    y_ordinal = np.zeros(n)
    for i in range(n):
        y_ordinal[i] = np.argmax(y[i])

    skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
    models = []
    fold_no = 0
    val_accuracies = []
    for train_index, val_index in skf.split(x, y_ordinal):
        fold_no = fold_no + 1
        model = create_model()
        model.fit(
            x=x[train_index],
            y=y[train_index],
            validation_data=(x[val_index], y[val_index]),
            epochs=NO_OF_EPOCHS,
            callbacks=callbacks,
        )
        # save trained model of the current fold
        models.append(model)
        # save validation accuracy of current fold
        predIdxs = model.predict(x[val_index])
        predIdxs = np.argmax(predIdxs, axis=1)
        y_ohe = y[val_index]
        val_classes = [
            np.argmax(y_ohe[i, :]) for i in range(y[val_index].shape[0])
        ]
        val_classes = np.array(val_classes)
        val_accuracy = accuracy_score(val_classes, predIdxs)
        val_accuracies.append(val_accuracy)

    # estimate of validation accuracy
    estimate_accuracy = np.mean(val_accuracies)
    return models, val_accuracies, estimate_accuracy


models, val_accuracies, estimate_accuracy = cross_validate_efficientnet()
print("estimated validation accuracy", estimate_accuracy)

best_model = models[np.argmax(val_accuracies)]
model = best_model
print("[INFO] evaluating after fine-tuning network head...")
predIdxs = model.predict(x=testGen, steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(
    classification_report(
        testGen.classes, predIdxs, target_names=testGen.class_indices.keys()
    )
)
