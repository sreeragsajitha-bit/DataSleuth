
try:
    from .dataset_loader import load_dataset
    from .dataset_profiler import profile_dataset
    from .data_quality import check_data_quality
    from .insight_engine import generate_insights
except ImportError:
    from dataset_loader import load_dataset
    from dataset_profiler import profile_dataset
    from data_quality import check_data_quality
    from insight_engine import generate_insights


def analyze_dataset(file_path):
    """
    Run the complete DataSleuth analysis pipeline.

    Steps:
    1. Load the dataset.
    2. Generate a dataset profile.
    3. Check data quality.
    4. Generate automated insights.

    Returns all results in one dictionary.
    """

    df = load_dataset(file_path)

    profile = profile_dataset(df)

    quality_issues = check_data_quality(df)

    insights = generate_insights(df)

    return {
        "dataset_name": str(file_path),
        "profile": profile,
        "quality_issues": quality_issues,
        "insights": insights
    }


def print_analysis_report(results):
    """Display a readable summary of the complete analysis."""

    profile = results["profile"]
    quality_issues = results["quality_issues"]
    insights = results["insights"]

    print("\n" + "=" * 55)
    print("              DATASLEUTH ANALYSIS REPORT")
    print("=" * 55)

    print(f"\nDataset: {results['dataset_name']}")

    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total rows: {profile['total_rows']}")
    print(f"Total columns: {profile['total_columns']}")
    print(f"Missing values: {profile['total_missing_values']}")
    print(f"Duplicate rows: {profile['duplicate_rows']}")

    print("\n2. COLUMN INFORMATION")
    print("-" * 40)

    print("Numerical columns:", profile["numerical_columns"])
    print("Categorical columns:", profile["categorical_columns"])
    print("Identifier columns:", profile["identifier_columns"])

    print("\n3. DATA QUALITY")
    print("-" * 40)

    if quality_issues:
        for issue in quality_issues:
            print(f"- {issue}")
    else:
        print("No data quality issues detected.")

    print("\n4. AUTOMATED INSIGHTS")
    print("-" * 40)

    if insights:
        for index, insight in enumerate(insights, start=1):
            print(
                f"\n{index}. [{insight['severity'].upper()}] "
                f"{insight['title']}"
            )
            print(insight["description"])
            print("Evidence:", insight["evidence"])
    else:
        print("No significant insights detected.")

    print("\n" + "=" * 55)
    print("             END OF DATASLEUTH REPORT")
    print("=" * 55)


if __name__ == "__main__":
    dataset_path = "data/raw/sales.csv"

    results = analyze_dataset(dataset_path)

    print_analysis_report(results)
