# Importing libraries
import kaggle
from pathlib import Path
import json
import pandas as pd
from Modules.load_data import load_data, merge_dfs

# URL for the dataset given by "username/data_set_name"
DATASET = "computingvictor/transactions-fraud-datasets"

# Absolute path to data folder
DATA_FOLDER = str(Path(__file__).parent.parent) + "/data"

# Obtaining absolute paths to relevant datasets

cards_data = DATA_FOLDER + "/cards_data.csv"
transaction_data = DATA_FOLDER + "/transactions_data.csv"


print(f"data folder path: {DATA_FOLDER}")
print(f"cards data path: {cards_data}")
print(f"transaction data path: {transaction_data}")
