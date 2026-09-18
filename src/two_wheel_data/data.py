import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter

DATASET = "emmanuelfwerr/motorcycle-technical-specifications-19702022"
RAW_FILE = "all_bikez_raw.csv"
CURATED_FILE = "all_bikez_curated.csv"
BRANDS_FILE = "bikez_brands.csv"


def load_bikes(file_path=RAW_FILE):
    return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, DATASET, file_path)


def load_processed(path=None):
    """The cleaned, parsed table. Build it first: python -m two_wheel_data.util.clean"""
    from two_wheel_data.util.clean import PROCESSED_PATH

    path = PROCESSED_PATH if path is None else path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - build it with: uv run python -m two_wheel_data.util.clean"
        )
    return pd.read_parquet(path)
