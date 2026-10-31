
import pandas as pd
import pytest

from insight_engine import (
    generate_insights,
    detect_missing_value_patterns,
    detect_numerical_outliers,
    detect_strong_correlations,
    detect_skewed_distributions,
    detect_categorical_imbalance,
    detect_constant_columns
)


def test_missing_value_detection():
    df = pd.DataFrame({"age": [20, None, 30, None]})
    findings = detect_missing_value_patterns(df)

    assert len(findings) == 1
    assert findings[0]["evidence"]["missing_count"] == 2
    assert findings[0]["evidence"]["missing_percentage"] == 50.0


def test_outlier_detection():
    df = pd.DataFrame({"sales": [1, 2, 3, 4, 100]})
    findings = detect_numerical_outliers(df)

    assert len(findings) == 1
    assert findings[0]["evidence"]["outlier_count"] == 1
    assert findings[0]["evidence"]["outlier_values"] == [100.0]


def test_correlation_detection():
    df = pd.DataFrame({
        "advertising": [1, 2, 3, 4, 5],
        "sales": [2, 4, 6, 8, 10]
    })
    findings = detect_strong_correlations(df)

    assert len(findings) == 1
    assert findings[0]["evidence"]["correlation"] == 1.0


def test_skewness_detection():
    df = pd.DataFrame({"income": [1, 1, 1, 1, 1, 10]})
    findings = detect_skewed_distributions(df)

    assert len(findings) == 1
    assert findings[0]["evidence"]["direction"] == "right"


def test_categorical_imbalance():
    df = pd.DataFrame({
        "gender": [
            "Male", "Male", "Male", "Male",
            "Male", "Male", "Male", "Male",
            "Female", "Female"
        ]
    })
    findings = detect_categorical_imbalance(df)

    assert len(findings) == 1
    assert findings[0]["evidence"]["dominant_value"] == "Male"
    assert findings[0]["evidence"]["dominant_percentage"] == 80.0


def test_constant_column_detection():
    df = pd.DataFrame({"status": ["Active", "Active", "Active"]})
    findings = detect_constant_columns(df)

    assert len(findings) == 1
    assert findings[0]["evidence"]["column"] == "status"


def test_clean_dataset():
    df = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "salary": [45000, 32000, 51000, 38000, 47000],
        "city": ["A", "B", "C", "D", "E"]
    })

    findings = generate_insights(df)

    assert findings == []


def test_identifier_columns_are_excluded_from_outliers():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 1000],
        "sales": [10, 20, 30, 40, 50]
    })
    findings = detect_numerical_outliers(df)

    detected_columns = [
        item["evidence"]["column"] for item in findings
    ]

    assert "customer_id" not in detected_columns


def test_empty_dataframe():
    df = pd.DataFrame()
    findings = generate_insights(df)

    assert findings == []


def test_invalid_input():
    with pytest.raises(TypeError):
        generate_insights([1, 2, 3])
