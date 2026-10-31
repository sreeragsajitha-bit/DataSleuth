
import pandas as pd

from data_quality import check_data_quality


def test_missing_values():
    df = pd.DataFrame({
        "name": ["Alex", "John"],
        "age": [20, None]
    })

    issues = check_data_quality(df)

    assert "age contains 1 missing values" in issues


def test_duplicate_rows():
    df = pd.DataFrame({
        "customer_id": [101, 101],
        "age": [20, 20]
    })

    issues = check_data_quality(df)

    assert "Dataset contains 1 duplicate rows" in issues


def test_negative_quantity():
    df = pd.DataFrame({
        "quantity": [2, -3]
    })

    issues = check_data_quality(df)

    assert "Quantity contains negative values" in issues


def test_negative_price():
    df = pd.DataFrame({
        "unit_price": [500, -200]
    })

    issues = check_data_quality(df)

    assert "Unit price contains negative values" in issues


def test_clean_dataset():
    df = pd.DataFrame({
        "customer_id": [101, 102],
        "quantity": [2, 3],
        "unit_price": [500, 1000]
    })

    issues = check_data_quality(df)

    assert issues == []
