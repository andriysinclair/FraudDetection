# Importing libraries
import kaggle
from pathlib import Path
import json
import pandas as pd
import argparse
import os
import yaml
import sys

# Obtaining Root dir
root = str(Path(__file__).parent.parent)
print(root)

# Add root to the Python path
sys.path.append(root)

from Modules.load_data import load_data, merge_dfs
from Modules.transforming import Target0_Reducer

# URL for the dataset given by "username/data_set_name"
DATASET = "computingvictor/transactions-fraud-datasets"

# Absolute path to data folder
DATA_FOLDER = root + "/data"

# Obtaining absolute paths to relevant datasets
cards_data_csv = DATA_FOLDER + "/cards_data.csv"
transaction_data_csv = DATA_FOLDER + "/transactions_data.csv"


# Instance of dataset reducer to obtain balanced df
transformer = Target0_Reducer(balanced=True)

# Obtaining global seed

# Load the config file
with open(root + "/config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config["global"]["seed"]

# print(f"seed: {seed}")


def main():
    parser = argparse.ArgumentParser(description="CLI tool to download & merge data.")

    parser.add_argument(
        "--download",
        action="store_true",
        help="Run command to download datasets from kaggle.",
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="run command to merge cards dataset with transactions dataset.",
    )

    parser.add_argument(
        "--reduce",
        action="store_true",
        help="run command to return reduced (balanced) df to use in final modelling.",
    )

    args = parser.parse_args()

    if args.download:

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

        else:
            print(
                "/data is not empty. Either data has already been installed or you need to clear the contents of the folder and try again"
            )

            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

    if args.merge:

        # Check that 'merged_data.pkl' is not in folder
        dir_list = [file for file in os.listdir(DATA_FOLDER)]
        if "merged_data.pkl" in dir_list:
            print(
                "files have already been merged. Remove 'merged_data.pkl' and try again."
            )
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

        elif len(dir_list) == 1:
            print("Files have not been downloaded.")
            print("Please run `frauddetection --download` then try again.")
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

        else:

            merge_dfs(
                transaction_data_csv=transaction_data_csv,
                cards_data_csv=cards_data_csv,
                data_folder=DATA_FOLDER,
            )

            print("Datasets successfully merged!")
            print("Please confirm that there is a file called: 'merged_data.pkl'\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

    if args.reduce:
        # check that balanced_data isn't in folder

        dir_list = [file for file in os.listdir(DATA_FOLDER)]
        if "balanced_data.pkl" in dir_list:
            print(
                "dataframe already exists. Remove 'balanced_data.pkl' and try again.\n"
            )
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

        elif "merged_data.pkl" not in dir_list:
            print("You have not merged the datasets yet.\n")
            print(
                "Run `frauddetection --download` to download. then `frauddetection --merge` to merge "
            )
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)

        else:
            print("Loading merged_data.pkl.\n")
            merged_df = pd.read_pickle(DATA_FOLDER + "/merged_data.pkl")

            print("Obtaining balanced dataframe.\n")
            balanced_df = transformer.fit_transform(merged_df)

            print("Dataframe obtained successfully. Loading to pickle")
            balanced_df.to_pickle(DATA_FOLDER + "/balanced_data.pkl\n")

            print("Successfully loaded dataframe")
            print("Displaying items inside /data...\n")

            for items in os.listdir(DATA_FOLDER):
                print(items)


if __name__ == "__main__":
    main()
