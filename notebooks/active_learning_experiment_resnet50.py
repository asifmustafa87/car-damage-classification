from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# set the matplotlib backend so figures can be saved in the background
import matplotlib
import keras

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython import display
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.uncertainty import entropy_sampling
from modAL.uncertainty import margin_sampling
from sklearn.metrics import classification_report, matthews_corrcoef
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import os

# Extracting the current base path
BASE_PATH = "./dataset/"

# define the names of the training, testing, and validation
# directories
TRAIN = "train"
TEST = "test"
VAL = "val"
# derive the paths to the training, validation,
# and testing directories
trainPath = os.path.sep.join([BASE_PATH, TRAIN])
valPath = os.path.sep.join([BASE_PATH, VAL])
testPath = os.path.sep.join([BASE_PATH, TEST])

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))
print(totalTrain, totalVal, totalTest)

# initialize the testing data augmentation object
valAug = ImageDataGenerator()

# initialize the testing generator
testGen = valAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=totalTest,
)
X_test, y_test = testGen.next()

pool_path = "./dataset/active_learning_data/"
poolAug = ImageDataGenerator()
# initialize the pool generator
poolGen = poolAug.flow_from_directory(
    pool_path,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=104,
)
X_pool, y_pool = poolGen.next()

trainAug = ImageDataGenerator()
BATCH_SIZE = 32
# initialize the training generator
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=547,
)
X_train, y_train = trainGen.next()


def random_sampling(classifier, X_pool):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    idx = np.ndarray(1)
    idx[0] = query_idx
    idx = idx.astype("int")

    query_idx = idx
    # return index of query and queried sample
    return query_idx, X_pool[query_idx]


# do not change this
# all needed sampling strategies

sampling_strategies = [
    random_sampling,
    entropy_sampling,
    margin_sampling,
    uncertainty_sampling,
]


# function that does an active learning run for 1 model
# includes 4 sampling stragies
def active_learning_run():
    sample_stragies_accuracy = []
    for sample_strategy in sampling_strategies:
        # load model
        model = load_model("./model/model.h5")
        model.make_predict_function()
        # cast model
        m = KerasClassifier(model, epochs=1)
        # initialize model
        m.initialize(X_test, y_test)
        # Extracting the current base path
        pool_path = "./dataset/active_learning_data/"
        # initialize the pool generator
        poolAug = ImageDataGenerator()
        poolGen = poolAug.flow_from_directory(
            pool_path,
            class_mode="categorical",
            target_size=(224, 224),
            color_mode="rgb",
            shuffle=False,
            batch_size=104,
        )
        # extract images in pool
        X_pool, y_pool = poolGen.next()
        # define active learner object
        learner = ActiveLearner(
            estimator=m,
            query_strategy=sample_strategy,
            X_training=X_train,
            y_training=y_train,
            verbose=3,
        )
        # initialize accuracy scores for 1 run
        accuracy_scores = [learner.score(X_test, y_test)]
        print(
            "Initial accuracy on test set before applying Active Learning",
            accuracy_scores,
        )

        for i in range(n_queries):
            display.clear_output(wait=True)
            query_idx, query_inst = learner.query(X_pool)
            print(query_inst.shape)
            # train model on new query
            learner.teach(query_inst, y_pool[query_idx])
            # delete query from pool
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), np.delete(
                y_pool, query_idx, axis=0
            )
            accuracy_scores.append(learner.score(X_test, y_test))
        sample_stragies_accuracy.append(accuracy_scores)
    return sample_stragies_accuracy


n_queries = 2
number_of_runs = 1
accuracy_scores_runs = []
for i in range(number_of_runs):
    print("run number", i + 1)
    accuracy = active_learning_run()
    accuracy_scores_runs.append(accuracy)

# create array for runs of random_samping
all_random_sampling_runs = np.zeros((number_of_runs, n_queries + 1))
# create array for runs of entropy_samping
all_entropy_sampling_runs = np.zeros((number_of_runs, n_queries + 1))
# create array for runs of margin_samping
all_margin_sampling_runs = np.zeros((number_of_runs, n_queries + 1))
# create array for runs of uncertainty_samping
all_uncertainty_sampling_runs = np.zeros((number_of_runs, n_queries + 1))
# array that combines all sampling strategy runs
all_sampling_runs = np.zeros((4 * number_of_runs, n_queries + 1))

for sample_strategy in sampling_strategies:
    index = sampling_strategies.index(sample_strategy)
    for i in range(number_of_runs):
        # print(len(accuracy_scores_runs[i][index]))
        all_sampling_runs[
            number_of_runs * index + i, :
        ] = accuracy_scores_runs[i][index]

# the next 4 lines assume
# sampling_strategies=[random_sampling,entropy_sampling,margin_sampling,uncertainty_sampling]
all_random_sampling_runs = all_sampling_runs[0:number_of_runs, :]
all_entropy_sampling_runs = all_sampling_runs[
    number_of_runs : 2 * number_of_runs, :
]

all_margin_sampling_runs = all_sampling_runs[
    2 * number_of_runs : 3 * number_of_runs, :
]
all_uncertainty_sampling_runs = all_sampling_runs[
    3 * number_of_runs : 4 * number_of_runs, :
]
# mean of random sampling runs
mean_random_sampling = np.mean(all_random_sampling_runs, axis=0)
# std of random sampling runs
std_random_sampling = np.std(all_random_sampling_runs, axis=0)
# mean of margin sampling runs
mean_margin_sampling = np.mean(all_margin_sampling_runs, axis=0)
# std of margin sampling runs
std_margin_sampling = np.std(all_margin_sampling_runs, axis=0)
# mean of entropy sampling runs
mean_entropy_sampling = np.mean(all_entropy_sampling_runs, axis=0)
# std of entropy sampling runs
std_entropy_sampling = np.std(all_entropy_sampling_runs, axis=0)
# mean of uncertainty sampling runs
mean_uncertainty_sampling = np.mean(all_uncertainty_sampling_runs, axis=0)
# std of uncertainty sampling runs
std_uncertainty_sampling = np.std(all_uncertainty_sampling_runs, axis=0)

mean = [
    mean_random_sampling.tolist(),
    mean_entropy_sampling.tolist(),
    mean_margin_sampling.tolist(),
    mean_uncertainty_sampling.tolist(),
]
std = [
    std_random_sampling.tolist(),
    std_entropy_sampling.tolist(),
    std_margin_sampling.tolist(),
    std_uncertainty_sampling.tolist(),
]

for i in range(4):
    mean[i][0] = 0.843
plt.figure(dpi=150)
n_queries = 4


def smooth(large, N):
    sum = 0
    result = list(0 for x in large)

    for i in range(0, N):
        sum = sum + large[i]
        result[i] = sum / (i + 1)

    for i in range(N, len(large)):
        sum = sum - large[i - N] + large[i]
        result[i] = sum / N

    return result


# plot entropy sampling
plt.plot(range(n_queries + 1), smooth(mean[1], 7), label="Entropy Sampling")
# plot least confidence sampling
plt.plot(range(n_queries + 1), smooth(mean[3], 7), label="Least Confidence")
# plot margin sampling
plt.plot(range(n_queries + 1), smooth(mean[2], 7), label="Margin Sampling")
# plot random sampling
plt.plot(range(n_queries + 1), smooth(mean[0], 7), label="Random Sampling")

plt.xlabel("Number of queries")
plt.ylabel("Accuracy")
plt.ylim([0.8, 0.90])
plt.title("Resnet50 Active Learning Results")
plt.grid()
plt.legend(loc="lower right")
