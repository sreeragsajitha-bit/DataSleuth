
import pandas as pd
from pathlib import Path


def load_dataset(file_path):

    path = Path(file_path)

    # Check whether the file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Check whether it is a CSV file
    if path.suffix.lower() != ".csv":
        raise ValueError(
            "Only CSV files are supported."
        )

    # Load CSV into a DataFrame
    df = pd.read_csv(path)

    return df


if __name__ == "__main__":

    file_path = "data/raw/sales.csv"

    df = load_dataset(file_path)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("---------------------------")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())
