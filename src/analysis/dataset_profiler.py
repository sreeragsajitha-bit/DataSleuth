
import pandas as pd


def profile_dataset(df):

    profile = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "statistics": df.describe().to_dict()
    }

    return profile


if __name__ == "__main__":

    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    result = profile_dataset(df)

    print("\n========== DATASET PROFILE ==========")

    print("Rows:", result["total_rows"])
    print("Columns:", result["total_columns"])
    print("Missing values:", result["total_missing_values"])
    print("Duplicate rows:", result["duplicate_rows"])

    print("\nColumn types:")
    print(result["column_types"])

    print("\nMissing values by column:")
    print(result["missing_values"])

    print("\n=====================================")
