
import pandas as pd

from dataset_profiler import profile_dataset


def test_total_rows():
    df = pd.DataFrame({"age": [20, 25, 30]})
    result = profile_dataset(df)
    assert result["total_rows"] == 3


def test_total_columns():
    df = pd.DataFrame({
        "age": [20, 25],
        "salary": [1000, 2000]
    })
    result = profile_dataset(df)
    assert result["total_columns"] == 2


def test_missing_values():
    df = pd.DataFrame({"age": [20, None, 30]})
    result = profile_dataset(df)
    assert result["total_missing_values"] == 1


def test_identifier_detection():
    df = pd.DataFrame({"customer_id": [101, 102, 103]})
    result = profile_dataset(df)
    assert "customer_id" in result["identifier_columns"]


def test_categorical_profiling():
    df = pd.DataFrame({"gender": ["Male", "Female", "Male"]})
    result = profile_dataset(df)
    profile = result["column_profiles"]["gender"]

    assert profile["unique_values"] == 2
    assert profile["most_frequent_values"][0]["value"] == "Male"


def test_outlier_detection():
    df = pd.DataFrame({"age": [20, 21, 22, 23, 100]})
    result = profile_dataset(df)

    assert result["column_profiles"]["age"]["statistics"]["outlier_count"] == 1


def test_correlation_detection():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10]
    })
    result = profile_dataset(df)

    assert len(result["high_correlations"]) == 1
    assert result["high_correlations"][0]["correlation"] == 1.0


def test_missing_percentage():
    df = pd.DataFrame({"age": [20, None, None, 30]})
    result = profile_dataset(df)

    assert result["column_profiles"]["age"]["missing_percentage"] == 50.0


def test_empty_dataframe():
    df = pd.DataFrame()
    result = profile_dataset(df)

    assert result["total_rows"] == 0
    assert result["total_columns"] == 0


def test_all_missing_column():
    df = pd.DataFrame({"age": [None, None, None]})
    result = profile_dataset(df)

    assert result["column_profiles"]["age"]["role"] == "unknown"
    assert result["column_profiles"]["age"]["missing_percentage"] == 100.0


def test_constant_column():
    df = pd.DataFrame({"quantity": [5, 5, 5, 5]})
    result = profile_dataset(df)

    stats = result["column_profiles"]["quantity"]["statistics"]

    assert stats["mean"] == 5.0
    assert stats["std"] == 0.0
    assert stats["outlier_count"] == 0


def test_single_row_dataset():
    df = pd.DataFrame({"age": [25], "quantity": [2]})
    result = profile_dataset(df)

    assert result["total_rows"] == 1
    assert result["correlation_matrix"] == {}
