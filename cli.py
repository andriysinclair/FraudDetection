# Importing libraries
import kaggle
from pathlib import Path
import json
import pandas as pd
import argparse
import os
from Modules.load_data import load_data, merge_dfs

# URL for the dataset given by "username/data_set_name"
DATASET = "computingvictor/transactions-fraud-datasets"

# Absolute path to data folder
DATA_FOLDER = str(Path(__file__).parent) + "/data"

# Obtaining absolute paths to relevant datasets

cards_data_csv = DATA_FOLDER + "/cards_data.csv"
transaction_data_csv = DATA_FOLDER + "/transactions_data.csv"


# print(f"data folder path: {DATA_FOLDER}")
# print(f"cards data path: {cards_data_csv}")
# print(f"transaction data path: {transaction_data_csv}")

# Function to check contents of /data folder


def main():
    parser = argparse.ArgumentParser(description="CLI tool to install data.")

    parser.add_argument(
        "--install",
        action="store_true",
        help="Run command to download datasets from kaggle and merge them",
    )

    args = parser.parse_args()

    if args.install:

        # Check that data folder is empty
        print(f"Intended path of data installation: {DATA_FOLDER}\n")
        print(f"Intended dataset to download: {DATASET}\n")

        if len(os.listdir(DATA_FOLDER)) == 1:
            print("/data is empty. Preparing to install...\n")

            load_data(dataset=DATASET, data_folder=DATA_FOLDER)

            print("Dataset has succesfully installed.")
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

            print("")
            print("Now merging datasets...\n")

            merge_dfs(
                transaction_data_csv=transaction_data_csv,
                cards_data_csv=cards_data_csv,
                data_folder=DATA_FOLDER,
            )

            print("Datasets successfully merged!")
            print("Please confirm that there is a file called: 'merged_data.pkl'\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

        else:
            print(
                "/data is not empty. Either data has already been installed or you need to clear the contents of the folder and try again"
            )
            print("Please do not remove the .gitkeep file.")
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)


if __name__ == "__main__":
    main()
