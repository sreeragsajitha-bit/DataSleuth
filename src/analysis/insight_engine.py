
import pandas as pd
from itertools import combinations


def create_finding(insight_type, severity, title, description, evidence):
    """Create a consistent, evidence-backed insight."""
    return {
        "type": insight_type,
        "severity": severity,
        "title": title,
        "description": description,
        "evidence": evidence
    }


def detect_missing_value_patterns(df):
    findings = []

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count == 0:
            continue

        missing_percentage = (
            missing_count / len(df) * 100
            if len(df) > 0 else 0
        )

        severity = (
            "high" if missing_percentage >= 50 else "medium"
        )

        findings.append(create_finding(
            insight_type="missing_values",
            severity=severity,
            title=f"Missing values in {column}",
            description=(
                f"Column '{column}' contains "
                f"{missing_count} missing values."
            ),
            evidence={
                "column": column,
                "missing_count": missing_count,
                "missing_percentage": round(missing_percentage, 2)
            }
        ))

    return findings


def detect_numerical_outliers(df):
    findings = []

    for column in df.select_dtypes(include="number").columns:
        if (
            column.lower() == "id"
            or column.lower().endswith("_id")
            or column.lower().endswith("_key")
        ):
            continue

        values = df[column].dropna()

        if len(values) < 4 or values.nunique() < 2:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = values[
            (values < lower_bound) |
            (values > upper_bound)
        ]

        if outliers.empty:
            continue

        percentage = len(outliers) / len(values) * 100

        findings.append(create_finding(
            insight_type="outliers",
            severity="high" if percentage >= 10 else "medium",
            title=f"Potential outliers in {column}",
            description=(
                f"Found {len(outliers)} potential outlier(s) "
                f"in column '{column}'."
            ),
            evidence={
                "column": column,
                "outlier_count": int(len(outliers)),
                "outlier_percentage": round(percentage, 2),
                "outlier_values": [
                    float(value) for value in outliers.tolist()
                ],
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }
        ))

    return findings


def detect_strong_correlations(df, threshold=0.7):
    findings = []

    numerical_df = df.select_dtypes(include="number")

    # Exclude identifier columns.
    valid_columns = [
        column for column in numerical_df.columns
        if not (
            column.lower() == "id"
            or column.lower().endswith("_id")
            or column.lower().endswith("_key")
        )
        and numerical_df[column].nunique(dropna=True) > 1
    ]

    if len(valid_columns) < 2 or len(df) < 2:
        return findings

    correlation_matrix = numerical_df[valid_columns].corr()

    for column_a, column_b in combinations(valid_columns, 2):
        correlation = correlation_matrix.loc[column_a, column_b]

        if pd.isna(correlation):
            continue

        if abs(correlation) >= threshold:
            direction = (
                "positive" if correlation > 0 else "negative"
            )

            findings.append(create_finding(
                insight_type="correlation",
                severity="medium",
                title=f"Strong correlation: {column_a} and {column_b}",
                description=(
                    f"'{column_a}' and '{column_b}' have a "
                    f"strong {direction} correlation."
                ),
                evidence={
                    "column_a": column_a,
                    "column_b": column_b,
                    "correlation": round(float(correlation), 4),
                    "direction": direction
                }
            ))

    return findings


def detect_skewed_distributions(df, threshold=1.0):
    findings = []

    for column in df.select_dtypes(include="number").columns:
        if (
            column.lower() == "id"
            or column.lower().endswith("_id")
            or column.lower().endswith("_key")
        ):
            continue

        values = df[column].dropna()

        if len(values) < 3 or values.nunique() < 2:
            continue

        skewness = values.skew()

        if pd.isna(skewness) or abs(skewness) < threshold:
            continue

        direction = "right" if skewness > 0 else "left"

        findings.append(create_finding(
            insight_type="skewness",
            severity="medium",
            title=f"Skewed distribution in {column}",
            description=(
                f"Column '{column}' has a {direction}-skewed "
                f"distribution."
            ),
            evidence={
                "column": column,
                "skewness": round(float(skewness), 4),
                "direction": direction,
                "threshold": threshold
            }
        ))

    return findings


def detect_categorical_imbalance(df, threshold=0.8):
    findings = []

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:
        values = df[column].dropna()

        if len(values) == 0 or values.nunique() < 2:
            continue

        frequencies = values.value_counts()
        top_value = frequencies.index[0]
        top_count = int(frequencies.iloc[0])
        top_percentage = top_count / len(values)

        if top_percentage >= threshold:
            findings.append(create_finding(
                insight_type="categorical_imbalance",
                severity="medium",
                title=f"Category imbalance in {column}",
                description=(
                    f"The value '{top_value}' dominates "
                    f"column '{column}'."
                ),
                evidence={
                    "column": column,
                    "dominant_value": str(top_value),
                    "dominant_count": top_count,
                    "dominant_percentage": round(
                        top_percentage * 100, 2
                    ),
                    "unique_categories": int(values.nunique())
                }
            ))

    return findings


def detect_constant_columns(df):
    findings = []

    for column in df.columns:
        values = df[column].dropna()

        if len(values) == 0 or values.nunique() != 1:
            continue

        constant_value = values.iloc[0]

        findings.append(create_finding(
            insight_type="constant_column",
            severity="low",
            title=f"Constant column: {column}",
            description=(
                f"Column '{column}' contains only one "
                f"distinct non-missing value."
            ),
            evidence={
                "column": column,
                "constant_value": str(constant_value),
                "unique_values": 1
            }
        ))

    return findings


def generate_insights(df):
    """
    Run all automated insight detectors on a DataFrame.

    Returns a list of evidence-backed findings.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    findings = []

    detectors = [
        detect_missing_value_patterns,
        detect_numerical_outliers,
        detect_strong_correlations,
        detect_skewed_distributions,
        detect_categorical_imbalance,
        detect_constant_columns
    ]

    for detector in detectors:
        findings.extend(detector(df))

    return findings


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")
    insights = generate_insights(df)

    print("\nDATA SLEUTH - AUTOMATED INSIGHTS")
    print("=" * 45)

    if not insights:
        print("No significant insights detected.")

    for index, insight in enumerate(insights, start=1):
        print(f"\n{index}. [{insight['severity'].upper()}] "
              f"{insight['title']}")

        print(insight["description"])
        print("Evidence:", insight["evidence"])
