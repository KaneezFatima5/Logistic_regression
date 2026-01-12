import os

CLASS_COLUMN = "class"  # The name of the class column
DROP_COLUMNS = [
    "id",
    "class"
]  # Any columns that should be dropped when training to prevent them from being split on
INDEX_COLUMN = "id"
CLASSIFIER = "random_forest" # choose from: lr, random_forest, gnb, and svm

DESIRED_SR = 44100
HPSS_MARGIN = 5.0
FRAME_LENGTH = 2048  # analysis window/frame
HOP_LENGTH = 512  # overlap b/w frames
N_MFCC = 13  # number of MFC co-efficients
N_FFT = 2048  # number of samples for each FFT (Fast Fourier Transform)

LR = 0.001 # learning rate
LAMBDA = 0.01  # Regression
EPOCHS = 5000  # no. of iteration
ADD_BIAS = True
KERNEL="rbf"
C=[0.01, 0.1, 1, 10, 100, 1000]  #regularization term for svm
GAMMA=[1, 0.1, 0.01, 0.001, 0.0001]  #influence term for svm

dirname = os.path.dirname(__file__)

TRAIN_MUSIC_PATH = os.path.join(
    dirname, "../data/music/train"
)  # The path to the training dataset
TEST_MUSIC_PATH = os.path.join(
    dirname, "../data/music/test"
)  # The path to the testing dataset

TRAIN_PATH = os.path.join(
    dirname, "../data/spreadsheets/train.csv"
)  # The path to the extracted train data
TEST_PATH = os.path.join(
    dirname, "../data/spreadsheets/test.csv"
)  # The path to the extracted test data
SUB_PATH = os.path.join(
    dirname, "../data/spreadsheets/sub.csv"
)  # The path where the submission dataset for kaggle will be stored
MODEL_PATH = os.path.join(
    dirname, "../models/lg.pkl"
)  # The path to where the model is stored to and loaded from
VISUALIZATIONS_PATH=os.path.join(
    dirname, "../data/visualizations"
)  # The path to where the visualizations are saved to

# Constants from Random Forest

SPLIT_FRAC = 0.01  # The fraction of instances being used for validating the trees.
SAMPLE_FRAC = 0.8  # The fraction of instances being bagged when training a decision tree in the random forest.
NUM_TREES = 31  # The number of decision trees to generate for the random forest, must be odd unless disabled in the random forest.
ALPHA = 0.05  # The alpha value used for chi-square related calculations
ACCURACY_THRESHOLD = (
    0.6  # The minimum accuracy of a tree required on the validation set.
)
FRAC_FEATURES = 0.5  # The fraction of features that are selected when training a decision tree in the random forest.
RANDOM_STATE = 2  # The random state used in various places.
