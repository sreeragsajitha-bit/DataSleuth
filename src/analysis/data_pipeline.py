
from dataset_loader import load_dataset
from dataset_profiler import profile_dataset
from data_quality import check_data_quality


def analyze_dataset(file_path):

    # Step 1: Load dataset
    df = load_dataset(file_path)

    # Step 2: Generate dataset profile
    profile = profile_dataset(df)

    # Step 3: Check data quality
    issues = check_data_quality(df)

    # Step 4: Return all results together
    return {
        "profile": profile,
        "quality_issues": issues
    }


if __name__ == "__main__":

    results = analyze_dataset("data/raw/sales.csv")

    profile = results["profile"]
    issues = results["quality_issues"]

    print("\n========== DATASLEUTH ANALYSIS ==========")

    print("\nDATASET OVERVIEW")
    print("Rows:", profile["total_rows"])
    print("Columns:", profile["total_columns"])
    print("Missing values:", profile["total_missing_values"])
    print("Duplicate rows:", profile["duplicate_rows"])

    print("\nDATA QUALITY REPORT")

    if issues:
        for issue in issues:
            print("-", issue)
    else:
        print("No data quality issues found.")

    print("\n=========================================")
