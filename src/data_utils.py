from pathlib import Path
from typing import Tuple
import pandas as pd

def get_data_directories() -> Tuple[str, str]:
    """ Returns the raw and preprocessed data directories.
    
    Returns:
        RAW_DATA_DIR (str): Directory for raw data.
        PRE_DATA_DIR (str): Directory for preprocessed data.
    """
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PRE_DATA_DIR = DATA_DIR / "preprocessed"

    error_message = lambda x: f"Unable to find data directory {x}, please check the project structure."

    assert DATA_DIR.exists(), error_message(DATA_DIR)
    assert RAW_DATA_DIR.exists(), error_message(RAW_DATA_DIR)
    assert PRE_DATA_DIR.exists(), error_message(PRE_DATA_DIR)

    print("Data directories successfully set.")

    return RAW_DATA_DIR, PRE_DATA_DIR

def save_dataset_to_csv(dataset: pd.DataFrame, dataset_filename: str):
    """ Saves dataset to a CSV file in the preprocessed data directory.
    
    Args:
        dataset (pandas.DataFrame): The dataset to be saved.
        dataset_filename (str): File name of the dataset when saved.
    """
    data_directories = get_data_directories()
    output_path = data_directories[1] / f"{dataset_filename}.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved in {data_directories[1]}")