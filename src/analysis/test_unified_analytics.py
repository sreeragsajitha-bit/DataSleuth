
import pandas as pd
import pytest

from unified_analytics import (
    validate_unified_plan,
    execute_unified_plan,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": [
            "Laptop",
            "Mouse",
            "Mouse",
            "Keyboard",
        ],
        "category": [
            "Electronics",
            "Accessories",
            "Accessories",
            "Accessories",
        ],
        "quantity": [1, 2, 3, 2],
        "unit_price": [55000, 400, 400, 1500],
    })


def test_filtered_revenue(sample_df):
    plan = {
        "filters": [
            {
                "column": "category",
                "operator": "eq",
                "value": "Accessories",
            },
            {
                "column": "quantity",
                "operator": "gt",
                "value": 1,
            },
        ],
        "logical_operator": "and",
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
    }

    result = execute_unified_plan(sample_df, plan)

    assert result["success"] is True
    assert result["original_row_count"] == 4
    assert result["filtered_row_count"] == 3
    assert result["query_result"]["result"] == 5000


def test_unfiltered_revenue(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
    }

    result = execute_unified_plan(sample_df, plan)

    assert result["filtered_row_count"] == 4
    assert result["query_result"]["result"] == 60000


def test_invalid_column_rejected(sample_df):
    plan = {
        "filters": [
            {
                "column": "unknown_column",
                "operator": "eq",
                "value": "Accessories",
            }
        ],
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
    }

    with pytest.raises(ValueError):
        validate_unified_plan(sample_df, plan)


def test_original_dataframe_unchanged(sample_df):
    original = sample_df.copy(deep=True)

    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
    }

    execute_unified_plan(sample_df, plan)

    pd.testing.assert_frame_equal(sample_df, original)


def test_unknown_plan_field_rejected(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "execute_python": "print('hello')",
    }

    with pytest.raises(ValueError):
        validate_unified_plan(sample_df, plan)
