# importing external libraries
from pathlib import Path
import os
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import warnings

# Set up logging configuration
warnings.filterwarnings("ignore")

set_config(transform_output = "pandas")

# Importing function to load data

from Modules.load_data import load_data
from Modules.preprocessing import missing_summary, dollar_to_int, find_unique_values
from Modules.plotting import Plotter
from Modules.transforming import *
from Modules.modelling import MLearner

# Importing Pipelines
from Modules.Pipelines import Pipeline1, Pipeline2, Pipeline3

# Obtaining Root dir

# Obtaining Root dir
root = str(Path(__file__).parent)
print(root)

# Obtaining seed from config.yaml

# Load the config file
with open(root + "/config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["global"]["seed"]

#print(f"seed: {seed}")

# Set global seeds for reproducibility
random.seed(seed)        
np.random.seed(seed)     

# Use the seed in scikit-learn
random_state = check_random_state(seed)

# Obtaining absolute path to data folder

data_folder = root + "/data"

# Loading the balanced data from pickle

balanced_df = pd.read_pickle(data_folder + "/balanced_data.pkl")

# Defining params for logit CV grid search

params_simple = {"penalty": [None], "solver":["saga"], "class_weight": ["balanced"], "max_iter":[1000]}

params_enhanced = [
    {"penalty": [None], "solver":["saga"], "class_weight": [None, "balanced"]},
    {"penalty": ["elasticnet"], "l1_ratio" : np.linspace(0,1,10).tolist(), 
     "C": np.linspace(0.01,1,10).tolist(), "solver":["saga"], "class_weight": ["balanced"], "max_iter":[1000]}    
]

# Obtaining score for no parameter opt

ML_pipe3 = MLearner(dataset=balanced_df, transformation_pipeline=Pipeline3, params=params_simple, estimator=LogisticRegression(random_state=seed), scoring="f1")
ML_pipe3.fit()
ML_pipe3.predict()

# Now hyperparameter tuning

ML_pipe3 = MLearner(dataset=balanced_df, transformation_pipeline=Pipeline3, params=params_enhanced, estimator=LogisticRegression(random_state=seed), scoring="accuracy")
ML_pipe3.fit()
ML_pipe3.predict()

# Now using LGBM

lgbm_param_grid = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Small values for smoother convergence
    "n_estimators": [250, 500, 1000],   # Large enough for early stopping to determine optimal rounds
    "num_leaves": [31, 50, 70],         # Control tree complexity; larger values for more complex patterns
    "min_child_weight": [0.001 ,1, 5, 10],     # Min data required in a child leaf
}


ML_pipe3 = MLearner(dataset=balanced_df, transformation_pipeline=Pipeline3, params=lgbm_param_grid, estimator=LGBMClassifier(random_state=seed, verbose=-1), scoring="f1",cv=5, verbose=0)
ML_pipe3.fit()
ML_pipe3.predict()
ML_pipe3.explain()