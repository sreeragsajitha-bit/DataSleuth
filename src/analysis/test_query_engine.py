
import pandas as pd
import pytest

from query_engine import (
    prepare_dataset,
    validate_query_plan,
    execute_query,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": [
            "Laptop",
            "Mouse",
            "Keyboard",
            "Laptop",
            "Mouse",
        ],
        "quantity": [1, 2, 1, 2, 3],
        "unit_price": [55000, 400, 1500, 55000, 400],
        "age": [21, 24, 23, 25, 22],
    })


def test_revenue_is_calculated(sample_df):
    result = prepare_dataset(sample_df)

    assert "revenue" in result.columns
    assert result["revenue"].tolist() == [
        55000, 800, 1500, 110000, 1200
    ]


def test_original_dataframe_is_not_modified(sample_df):
    original_columns = sample_df.columns.tolist()

    prepare_dataset(sample_df)

    assert sample_df.columns.tolist() == original_columns


def test_total_revenue(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
    }

    result = execute_query(sample_df, plan)

    assert result["result"] == 168500


def test_average_quantity(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "mean",
    }

    result = execute_query(sample_df, plan)

    assert result["result"] == 1.8


def test_revenue_grouped_by_product(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
    }

    result = execute_query(sample_df, plan)

    revenue_by_product = {
        row["product"]: row["revenue"]
        for row in result["result"]
    }

    assert revenue_by_product == {
        "Laptop": 165000,
        "Mouse": 2000,
        "Keyboard": 1500,
    }


def test_top_three_products(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 3,
    }

    result = execute_query(sample_df, plan)

    assert result["result"][0]["product"] == "Laptop"
    assert result["result"][0]["revenue"] == 165000


def test_count_aggregation(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "count",
    }

    result = execute_query(sample_df, plan)

    assert result["result"] == 5


def test_median_aggregation(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "median",
    }

    result = execute_query(sample_df, plan)

    assert result["result"] == 2


def test_invalid_aggregation(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "execute",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_unknown_metric(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "salary",
        "aggregation": "sum",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_unknown_group_column(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "sum",
        "group_by": "department",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_arbitrary_execution_field_is_rejected(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "sum",
        "python_code": "import os",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_invalid_limit(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "sum",
        "limit": -1,
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_mean_on_non_numeric_column(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "product",
        "aggregation": "mean",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_invalid_sort_order(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "sum",
        "sort_order": "sideways",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_invalid_operation(sample_df):
    plan = {
        "operation": "delete",
        "metric": "quantity",
        "aggregation": "sum",
    }

    with pytest.raises(ValueError):
        execute_query(sample_df, plan)


def test_validate_query_plan(sample_df):
    plan = {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "mean",
    }

    assert validate_query_plan(sample_df, plan) is True
