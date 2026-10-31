
def check_data_quality(df):

    issues = []

    # Check for missing values
    missing = df.isnull().sum()

    for column, count in missing.items():
        if count > 0:
            issues.append(
                f"{column} contains {count} missing values"
            )

    # Check for duplicate rows
    duplicates = df.duplicated().sum()

    if duplicates > 0:
        issues.append(
            f"Dataset contains {duplicates} duplicate rows"
        )

    # Check for negative quantities
    if "quantity" in df.columns:
        if (df["quantity"] < 0).any():
            issues.append(
                "Quantity contains negative values"
            )

    # Check for negative prices
    if "unit_price" in df.columns:
        if (df["unit_price"] < 0).any():
            issues.append(
                "Unit price contains negative values"
            )

    return issues


if __name__ == "__main__":

    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    issues = check_data_quality(df)

    print("\n========== DATA QUALITY REPORT ==========")

    if issues:
        for issue in issues:
            print("-", issue)
    else:
        print("No data quality issues found.")

    print("=========================================")
