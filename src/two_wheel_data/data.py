import kagglehub
from kagglehub import KaggleDatasetAdapter

DATASET = "emmanuelfwerr/motorcycle-technical-specifications-19702022"
RAW_FILE = "all_bikez_raw.csv"
CURATED_FILE = "all_bikez_curated.csv"
BRANDS_FILE = "bikez_brands.csv"


def load_bikes(file_path=RAW_FILE):
    return kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, DATASET, file_path)
