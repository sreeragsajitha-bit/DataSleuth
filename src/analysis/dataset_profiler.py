
import pandas as pd
from itertools import combinations


def is_identifier(column_name):
    name = column_name.lower()
    return (
        name == "id"
        or name.endswith("_id")
        or name.endswith("_key")
    )


def identify_column_type(series, column_name):
    if is_identifier(column_name):
        return "identifier"

    if series.isna().all():
        return "unknown"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    return "categorical"


def profile_numerical_column(series):
    clean_series = series.dropna()

    if clean_series.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q1": None,
            "q3": None,
            "outlier_count": 0,
            "outlier_lower_bound": None,
            "outlier_upper_bound": None
        }

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (clean_series < lower_bound) |
        (clean_series > upper_bound)
    ).sum()

    return {
        "count": int(clean_series.count()),
        "mean": float(clean_series.mean()),
        "median": float(clean_series.median()),
        "std": (
            float(clean_series.std())
            if len(clean_series) > 1 else None
        ),
        "min": float(clean_series.min()),
        "max": float(clean_series.max()),
        "q1": float(q1),
        "q3": float(q3),
        "outlier_count": int(outlier_count),
        "outlier_lower_bound": float(lower_bound),
        "outlier_upper_bound": float(upper_bound)
    }


def profile_categorical_column(series):
    clean_series = series.dropna()

    return {
        "unique_values": int(clean_series.nunique()),
        "most_frequent_values": [
            {
                "value": str(value),
                "count": int(count)
            }
            for value, count in clean_series.value_counts().head(5).items()
        ]
    }


def calculate_correlations(df, numerical_columns):
    if len(numerical_columns) < 2:
        return {}, []

    valid_columns = [
        column for column in numerical_columns
        if df[column].nunique(dropna=True) > 1
    ]

    if len(valid_columns) < 2:
        return {}, []

    correlation_df = df[valid_columns].corr(method="pearson")
    correlation_matrix = correlation_df.to_dict()

    high_correlations = []

    for col_a, col_b in combinations(valid_columns, 2):
        correlation = correlation_df.loc[col_a, col_b]

        if pd.notna(correlation) and abs(correlation) >= 0.7:
            high_correlations.append({
                "column_a": col_a,
                "column_b": col_b,
                "correlation": float(correlation)
            })

    return correlation_matrix, high_correlations


def profile_dataset(df):
    column_profiles = {}

    numerical_columns = []
    categorical_columns = []
    identifier_columns = []
    outlier_summary = {}

    for column in df.columns:
        series = df[column]

        column_type = identify_column_type(series, column)

        missing_count = int(series.isna().sum())
        missing_percentage = (
            float((missing_count / len(df)) * 100)
            if len(df) > 0 else 0.0
        )

        profile = {
            "data_type": str(series.dtype),
            "role": column_type,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage
        }

        if column_type == "numerical":
            numerical_columns.append(column)

            statistics = profile_numerical_column(series)
            profile["statistics"] = statistics

            profile["unique_values"] = int(series.nunique())
            outlier_summary[column] = statistics["outlier_count"]

        elif column_type == "categorical":
            categorical_columns.append(column)
            profile.update(profile_categorical_column(series))

        elif column_type == "identifier":
            identifier_columns.append(column)
            profile["unique_values"] = int(series.nunique())

        elif column_type == "unknown":
            profile["statistics"] = {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "q1": None,
                "q3": None,
                "outlier_count": 0,
                "outlier_lower_bound": None,
                "outlier_upper_bound": None
            }

        column_profiles[column] = profile

    correlation_matrix, high_correlations = calculate_correlations(
        df, numerical_columns
    )

    statistics = (
        df.describe().to_dict()
        if len(df.columns) > 0 else {}
    )

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "statistics": statistics,
        "column_profiles": column_profiles,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "identifier_columns": identifier_columns,
        "correlation_matrix": correlation_matrix,
        "high_correlations": high_correlations,
        "outlier_summary": outlier_summary
    }
