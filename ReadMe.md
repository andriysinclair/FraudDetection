# LoanPrediction Project

## Project Overview
This project focuses on building a pipeline to detect fraudulent transactions in loan-related datasets. It involves data preprocessing, exploratory data analysis (EDA), feature engineering, and model training.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup and Installation](#setup-and-installation)
3. [Repo Description](#repo-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Training and Pipelines](#model-training-and-pipelines)
6. [Results and Explanatory Analysis](#results-and-visualizations)

---

## Prerequisites

1. Python 3.11 or newer installed.
2. [Conda](https://anaconda.org/anaconda/conda) installed (e.g., Anaconda or Miniconda).
2. Install `pip` (Python's package installer), which is included in Conda installations.
3. A Kaggle API key saved as `~/.kaggle/kaggle.json`. [Learn how to create a Kaggle API key](https://www.kaggle.com/docs/api).

## Setup and Installation

To set up the environment and dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/andriysinclair/FraudDetection
   cd FraudDetection

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate FraudDetection

3. Install the package
   ```bash
   pip install -e

4. Download the dataset
   ```bash
   fraudetection --install

---

## Repo Description
```plaintext
.
├── cli.py                          # CLI entry point for downloading datasets of kaggle and merging them.
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
├── modelling.ipynb                 # Model selection and hyper-parameter tuning notebook
├── Modules                         # Modules folder
│   ├── __init__.py                 # Marks folder a Python package
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
│   └── merch_city_barplot.pdf      # Fraudulent transactions per merchant city
├── pyproject.toml                  # Build configuration and metadata for the project
├── ReadMe.md                       # Documentation for the repository
└── Tests                           # Unit tests for custom transformations

---
