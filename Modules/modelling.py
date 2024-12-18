# importing external libraries
from pathlib import Path
import os
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

# Obtaining Root dir

root = str(Path(__file__).parent.parent)

# Obtaining seed from config.yaml

# Load the config file
with open(root + "/config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["global"]["seed"]


class MLearner:
    """

    Class to streamline pipeline transformation, hyperparameter tuning, training, testing and CV
    """

    def __init__(
        self,
        dataset,
        transformation_pipeline,
        params,
        estimator,
        scoring="f1",
        cv=5,
        HPT=True,
        target_col="is_fraud",
    ):
        """
        Args:
            dataset (pd.DataFrame): raw untrnasformed (but merged) dataframe.
            transformation_pipeline (sklearn.pipeline.Pipeline): Transformation Pipeline from Pipelines.py
            params (dict): Params to GridSearch over - if single params are given - this class defaults to regular training and testing with CV
            estimator (sklearn.base.BaseEstimator): sklearn estimator to use
            scoring (str, optional): evaluation metric. Defaults to "f1".
            cv (int, optional): cross-validation sets. Defaults to 5.
        """
        self.dataset = dataset
        self.transformation_pipeline = transformation_pipeline
        self.params = params
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.target_col = target_col
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.grid_searcher = None

    def fit(self):
        """fit

        Fits to training data
        """

        # Test Train Split

        X_train, X_test = train_test_split(self.dataset, random_state=seed)

        # Applying Pipeline1

        X_train = self.transformation_pipeline.fit_transform(X_train)
        X_test = self.transformation_pipeline.transform(X_test)

        # Obtaining target column

        y_train = X_train[self.target_col]
        y_test = X_test[self.target_col]

        self.y_train = y_train
        self.y_test = y_test

        # Dropping is fraud column

        X_train = X_train.drop(columns=[self.target_col])
        X_test = X_test.drop(columns=[self.target_col])

        self.X_train = X_train
        self.X_test = X_test

        # Checkiing the proportion of positive values

        print(f"% of fraudulent transactions in y_train: {y_train.mean()}")
        print(f"% of fraudulent transactions in y_test: {y_test.mean()}\n")

        grid_searcher = GridSearchCV(
            self.estimator,
            param_grid=self.params,
            scoring=self.scoring,
            cv=self.cv,
            verbose=3,
        )

        self.grid_searcher = grid_searcher

        grid_searcher.fit(X_train, y_train)

        # Print the best parameters
        print(f"Best parameters found: {grid_searcher.best_params_}")

        # This class can be used as

    def predict(self):
        """predict

        Prints score of estimator (with best parameters) on the training set and the testing set.
        """
        print(
            f"score on training set: {self.grid_searcher.score(self.X_train, self.y_train)}"
        )
        print(
            f"score on testing set: {self.grid_searcher.score(self.X_test, self.y_test)}"
        )
