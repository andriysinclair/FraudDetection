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
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import dalex as dx
import kaleido

# Obtaining Root dir

root = str(Path(__file__).parent.parent)

# Add root to the Python path
sys.path.append(root)

# Importing function to load data

from Modules.plotting import Plotter

# Obtaining plot folder
plot_folder = root + "/Plots"

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
        verbose=1,
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
        self.verbose = verbose

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
            verbose=self.verbose,
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
        y_pred = self.grid_searcher.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        print("")
        print("Performance on test set with best parameters: ")

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1 Score: {f1:.2f}")

    def explain(self):

        # Ensure the model is fitted
        if not self.grid_searcher:
            raise ValueError("The model has not been fitted. Call 'fit' first.")

        # get the best estimator
        best_model = self.grid_searcher.best_estimator_

        # Check for feature_importances_ attribute
        if hasattr(best_model, "feature_importances_"):

            fi = best_model.feature_importances_
            # print(fi)
            cols = self.X_test.columns
            # print(cols)

            fi_df = pd.DataFrame({"Features": cols, "Feature Importance": fi})
            # print(fi_df)

            plotter = Plotter(df=fi_df)

            plotter.bar_plot(
                feature_of_interest="Features",
                top_n=len(cols),
                Plots_folder=plot_folder,
                file_name="fi_plots",
                target="Feature Importance",
                ylabel="Feature Importance",
                title="Column by Feature Importance",
            )

            # PDP plots

            # create an explainer class
            lgbm_exp = dx.Explainer(self.grid_searcher, self.X_train, self.y_train)

            # PDP profile for transaction amount and credit limit
            lgbm_profile = lgbm_exp.model_profile(
                variables=["amount", "credit_limit"], type="partial"
            )

            pdp_data = lgbm_profile.result

            # Apply inverse transform to numerical columns

            # Obtaining encoder for amount and credit_limit
            encoder = self.transformation_pipeline.named_steps[
                "exploration_pipeline"
            ].named_steps["Encoder"]
            scaler_amount = encoder.named_transformers_["r"]
            scaler_credit_limit = encoder.named_transformers_["s"]

            # Apply inverse transform to numerical columns in PDP
            for variable, scaler in zip(
                ["amount", "credit_limit"], [scaler_amount, scaler_credit_limit]
            ):
                variable_data = pdp_data[pdp_data["_vname_"] == variable].copy()
                variable_data["_x_"] = scaler.inverse_transform(
                    variable_data["_x_"].values.reshape(-1, 1)
                ).flatten()

                plt.figure(figsize=(8, 6))
                plt.plot(
                    variable_data["_x_"],
                    variable_data["_yhat_"],
                    label=f"PDP for {variable}",
                )
                plt.title(f"Partial Dependence Plot for {variable}")
                plt.xlabel(variable + " ($)")
                plt.ylabel("Predicted Outcome")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                # Save plot to the specified folder
                plt.savefig(plot_folder + f"/{variable}.pdf", format="pdf")
                plt.show()

        else:
            raise ValueError("The estimator does not support feature importance.")
