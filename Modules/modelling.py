# importing external libraries
from pathlib import Path
import os
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class MLearner:

    def __init__(
        self, dataset, transformation_pipeline, params, estimator, scoring="f1", cv=5
    ):
        self.dataset = dataset
        self.transformation_pipeline = transformation_pipeline
        self.params = params
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.y_train = None
        self.y_test = None
        self.grid_searcher = None

    def fit(self):
        # Test Train Split

        X_train, X_test = train_test_split(self.dataset)

        # Applying Pipeline1

        X_train = self.transformation_pipeline.fit_transform(X_train1)
        X_test = self.transformation_pipeline.transform(X_test1)

        # Obtaining target column

        y_train = X_train["is_fraud"]
        y_test = X_test["is_fraud"]

        self.y_train = y_train
        self.y_test = y_test

        # Dropping is fraud column

        X_train = X_train.drop(columns=["is_fraud"])
        X_test = X_test.drop(columns=["is_fraud"])

        # Checkiing the proportion of positive values

        print(f"% of fraudulent transactions in y_train: {y_train.mean()}")
        print(f"% of fraudulent transactions in y_test: {y_test.mean()}\n")

        grid_searcher = GridSearchCV(
            self.estimator, param_grid=self.params, scoring=self.scoring, cv=self.cv
        )

        self.grid_searcher = grid_searcher

        grid_searcher.fit(X_train, y_train)

    def predict(self):
        pass
