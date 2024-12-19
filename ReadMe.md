# LoanPrediction Project

## Project Overview
This project focuses on building a pipeline to detect fraudulent transactions in loan-related datasets. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, and model training.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup and Installation](#setup-and-installation)
3. [Repo Description](#repo-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Training, Pipelines and Results](#model-training-pipelines-and-results)

---

## Prerequisites

1. Python 3.12 or newer installed.
2. [Conda](https://anaconda.org/anaconda/conda) installed (e.g., Anaconda or Miniconda).
2. Install `pip` (Python's package installer), which is included in Conda installations.
3. A Kaggle API key saved as `~/.kaggle/kaggle.json`. [Learn how to create a Kaggle API key](https://www.kaggle.com/docs/api).

---

## Setup and Installation

To set up the environment and dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/andriysinclair/FraudDetection
   cd FraudDetection
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate FraudDetection
   ```

3. Install the package
   ```bash
   pip install -e .
   ```

4. Load the dataset
   - To download the individual files from Kaggle:
     ```bash
     frauddetection --download
     ```

   - To merge the cards and transactions files:
     **WARNING:** These are large datasets and will most likely require a computing cluster to compute.
     Requires individual csv's to exist in datafolder.
     ```bash
     frauddetection --merge
     ```

   - To return a reduced and balanced data frame where the number of fraudulent transactions equals the number of non-fraudulent transactions:
     This dataset is of a workable size and is included in the installation.
     Requires `merged_data.pkl` to exist in data folder.
     ```bash
     frauddetection --reduce
     ```

---

## Repo Description
```plaintext
.
├── config.yaml                     # Configuration file for project settings
├── data                            # Contains raw and processed data files (before running
                                      frauddetection --install this will only contain .gitkeep)
│   ├── cards_data.csv              # cards data
│   ├── mcc_codes.json              # Merchant Category Codes (MCC) mapping
│   ├── merged_data.pkl             # Complete dataset for analysis, appears after frauddetection --install
│   ├── train_fraud_labels.json     # fraudulent/ non-fraudulent Yes/ No mapping
│   ├── transactions_data.csv       # Transactions data
│   └── users_data.csv              # Users data (not used)
├── eda_cleaning.ipynb              # Notebook for exploration and visualisations
├── environment.yaml                # Conda environment config file
├── model_selection.ipynb           # Model selection and hyper-parameter tuning notebook for large dataset
├── model_training.ipynb            # Model selection and hyper-parameter tuning notebook for balanced dataset
├── Modules                         # Modules folder
    ├── __init__.py                 # Marks folder a Python package
    ├── cli.py                      # CLI entry point for downloading datasets of kaggle and merging them.
│   ├── load_data.py                # Functions to download and merge datasets
│   ├── modelling.py                # Class for hyper-parameter tuning, training, testing and CV.
│   ├── Pipelines.py                # Data transformation pipelines
│   ├── plotting.py                 # Plotting class
│   ├── preprocessing.py            # Some pre-processing and exploration functions
│   └── transforming.py             # Custom Sklearn transformation classes: target encoding, data decomp etc.
├── Plots                           # Folder with EDA plots
│   ├── amount_box_plot.pdf         # Distribution of transaction amount by fraud status
│   ├── corr_plot1.pdf              # Correlations of features with target variable (no date decomp)
│   ├── date_cols_corrs.pdf         # Correlations of date decomp features with target variable.
│   ├── mcc_barplot.pdf             # Fraudulent transactions per MCC
│   └── amount.pdf                  # PDP of amount
│   └── credit_limit.pdf            # PDP of credit_limit
│   └── fi_plots.pdf                # Feature importance plots of LGBM
│   └── merch_city_barplot.pdf      # Fraudulent transactions per merchant city
├── pyproject.toml                  # Build configuration and metadata for the project
├── ReadMe.md                       # Documentation for the repository
└── Tests                           # Unit tests for custom transformations
    ├── __init__.py                 # Marks folder a Python package
    └── tests_transforming.py       # Unit tests for custom transformers
```
---

## Exploratory Data Analysis (EDA)

**Found in `eda_cleaning.ipynb`.**

*Please note that notebook was designed to be used with `merged_data.pkl`. Access to a computing cluster and the necessary .csv files are required to obtain this. Please run the commands in the following order:*
1. `$ frauddetection --download`
2. `$ frauddetection --merge`

*If this is is not possible, notebook also works with `balanced_data.pkl` (Please unhash the relevant code snippit). Please keep in mind that results may be slightly different*

This notebook contains the following:
* info on columns: type, missing values, unique values
* Distribution of target column
* plots on:
    - Correlations of features with target: `Plots/corr_plot1.pdf` & `Plots/date_cols_corrs.pdf`
    - Fraudulent transactions by MCC: `Plots/mcc_barplot.pdf`
    - Fruadulent transactions by merchant city: `Plots/merch_city_barplot.pdf`

--- 

## Model Training, Pipelines and Results

*Custom scikit-learn classes can be found in: `Modules/transforming.py`. These are used to make pipelines, which are found in `Modules/Pipelines.py`. Unit tests on the `Target0_Reducer` and `CustomTargetEncoder` classes can be ran by (from the root directory):*
1. `$ cd tests`
2. `$ pytest tests_transforming.py`

## Custom transformers
*found in `Modules/transforming.py`*

### `TargetBinary`
- Converts a column or DataFrame with categorical "Yes"/"No" values into binary (1/0).

### `Date`
- Converts object-type date columns to `datetime64[ns]` format and extracts relevant components.

### `DateDecomposer`
- Extracts granular components (e.g., hour, day of the week, month, year) from date-like features.

### `Target0_Reducer`
- Reduces the size of non-fraudulent samples to balance an imbalanced dataset, maintaining the target's proportions or equalizing class sizes.

### `CustomTargetEncoder`
- Encodes categorical features using the target mean calculated from the training set.

### `DollarToInt`
- Converts dollar-formatted strings into integers for numerical processing.

### `TimeSeriesMapper`
- Maps date-like columns to a sequential time series index.

### `RemoveUncorrFeatures`
- Removes features with a correlation below a specified threshold with the target variable.

## Pipelines
*found in `Modules/Pipelines.py`*

## Auxiliary Functions

### `fillna_transformer`
- Fills missing values in a dataset with `0` using `FunctionTransformer`.

### `copy_target_column`
- Copies the `target` column and creates a new column `is_fraud` for use in pipelines.

### `rename_cols_transformer`
- Renames transformed columns to cleaner names after applying column transformations.

### `time_series_pipeline`
- Converts date-like features into `datetime64` format and creates ordered time series mappings.

---

## Pipelines

### **General Transformation Pipelines**
Reusable pipelines for transforming and reducing datasets.

1. **`general_transformation_pipeline1`**
   - Balances the dataset by selecting 1% of non-fraudulent samples.
   - Applies transformations for numerical and date-like features.
   - Adds binary transformations for categorical features like `has_chip` and `card_on_dark_web`.

2. **`general_transformation_pipeline2`**
   - Includes more granular decomposition of date-like features (e.g., extracting hour, day, month, and year).
   - Target encodes features for downstream analysis.

3. **`general_transformation_pipeline3`**
   - Similar to Pipeline 2 but allows varying proportions of non-fraudulent samples to be used.

---

### **Exploration Pipelines**
Enhances data exploration through encoding, scaling, and feature engineering.

1. **`Pipeline_for_exploration1`**
   - Adds encoding for categorical features like `use_chip` and `card_brand`.
   - Scales numerical features (`amount` and `credit_limit`).

2. **`Pipeline_for_exploration2`**
   - Decomposes date-like features into granular components (e.g., hour, day of the week).
   - Applies target encoding to additional features like `date_hour`, `expires_month`, and `acct_open_date_year`.

3. **`Pipeline_for_exploration3`**
   - Similar to Pipeline 2 but allows varying proportions of non-fraudulent samples to be used.

---

### **Final Pipelines**

1. **`Pipeline1`**
   - Combines time-series processing and exploration pipelines.
   - Balances the dataset with 1% of non-fraudulent samples.
   - Removes uncorrelated features with a threshold of 0.05.

2. **`Pipeline2`**
   - Adds granular decomposition of date-like features and target encoding.
   - Balances the dataset with 1% of non-fraudulent samples.
   - Removes uncorrelated features with a threshold of 0.05.

3. **`Pipeline3`**
   - Similar to `Pipeline2` but allows for flexible sampling proportions of non-fraudulent samples.
   - Applies target encoding and removes uncorrelated features.
---

## MLearner class
*found in `Modules/modelling.py`*

The `MLearner` class streamlines machine learning workflows, including preprocessing, hyperparameter tuning, training, evaluation, and explainability.

### Key Features:
- Train-test splitting and pipeline transformations.
- Hyperparameter tuning with `GridSearchCV`.
- Performance metrics: Accuracy, Precision, Recall, F1 Score.
- Feature importance and Partial Dependence Plots (PDPs).

### Usage:
1. Initialize with dataset, pipeline, model, and parameters.
2. Call `fit()` to train and tune the model.
3. Use `predict()` for evaluation and `explain()` for insights.

## Results

**`model_selection.iynb`**
*This is designed to work with the full `merged_data.pkl`. The purpouse of this notebook was to identify whether `Pipeline1` had better performance than `Pipeline2` and if increasing the amount of non-fraudulent samples would increase performance (at what computation cost).*

**`model_training.py`**
*Carries out hyper parameter tuning for GLM (logit) and LGBM models and uses optimised paramters on testing set. Obtains feature importance and PDP plots for `amount` and `credit_limit`. Saved to:*
* `Plots/fi_plots.pdf`
* `Plots/amount.pdf`
* `Plots/credit_limit.pdf`

