
import pandas as pd

from dataset_profiler import profile_dataset


def test_total_rows():
    df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "age": [20, 21, 22]
    })

    result = profile_dataset(df)

    assert result["total_rows"] == 3


def test_total_columns():
    df = pd.DataFrame({
        "name": ["A", "B"],
        "age": [20, 21]
    })

    result = profile_dataset(df)

    assert result["total_columns"] == 2


def test_missing_values():
    df = pd.DataFrame({
        "name": ["A", "B", "C"],
        "age": [20, None, 22]
    })

    result = profile_dataset(df)

    assert result["total_missing_values"] == 1
