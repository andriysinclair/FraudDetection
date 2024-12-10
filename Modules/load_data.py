# Please ensure your kaggle.json file (with kaggle API key) is in the correct folder
# ~/.kaggle/kaggle.json for LINUX

# Importing libraries
import kaggle
from pathlib import Path


# URL for the dataset given by "username/data_set_name"
dataset = "computingvictor/transactions-fraud-datasets"

# Absolute path to data folder
data_folder = Path(__file__).parent.parent / "data"


def load_data(dataset=dataset, data_folder=data_folder):

    # Automatically looks for the kaggle.json file and authenticates user
    kaggle.api.authenticate()

    # Downloading dataset ...
    kaggle.api.dataset_download_files(dataset, path=data_folder, unzip=True)


print(data_folder)
