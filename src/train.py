#!/usr/bin/env python3

from pathlib import Path
import time

from utils import *
from config import *
import logistic_regression
import svm
import gaussian_naive_bayes
import random_forest

def train(train_path, save_path, classifier):
    """Train a random forest using the config file's hyperparameters given the path to the training data and save location."""
    train_df = load_spreadsheet(train_path)
    print("Training model")
    start_time = time.time()
    model=None
    if classifier=="lr":
        print("Using logistic regression")
        model = logistic_regression.LogisticRegression(
            train_df,
            CLASS_COLUMN,
            DROP_COLUMNS,
            ADD_BIAS
        )
        model.train(train_df, CLASS_COLUMN, LR, LAMBDA, EPOCHS)
    elif classifier=="random_forest":
        print("Using Random Forest")
        model=random_forest.RandomForest()
        model.train(train_df)
    elif classifier=="gnb":
        print("Gaussian Naive Bayes")
        model=gaussian_naive_bayes.GaussianNaiveBayes()
        model.train(train_df)
    else:
        print("Support Vector Machines")
        model=svm.SupportVectorMachine()
        model.train(train_df)

    end_time = time.time()
    print(f"Training took: {end_time - start_time} seconds")

    print("Saving model")
    save_model(save_path, model)


if __name__ == "__main__":
    train_loc = Path(TRAIN_PATH)
    save_loc = Path(MODEL_PATH)
    train(train_loc, save_loc, CLASSIFIER)
