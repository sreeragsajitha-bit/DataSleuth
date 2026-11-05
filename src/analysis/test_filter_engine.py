
import pandas as pd
import pytest

from filter_engine import (
    validate_filters,
    apply_filters,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": [
            "Laptop",
            "Mouse",
            "Keyboard",
            "Laptop",
            "Monitor",
        ],
        "category": [
            "Electronics",
            "Accessories",
            "Accessories",
            "Electronics",
            "Electronics",
        ],
        "quantity": [1, 2, 1, 3, 2],
        "unit_price": [55000, 400, 1500, 55000, 12000],
        "age": [21, 24, 23, 28, None],
    })


def test_filter_equal(sample_df):
    filters = [
        {"column": "product", "operator": "eq", "value": "Laptop"}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 2
    assert all(result["product"] == "Laptop")


def test_filter_not_equal(sample_df):
    filters = [
        {"column": "product", "operator": "ne", "value": "Laptop"}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 3
    assert "Laptop" not in result["product"].values


def test_filter_greater_than(sample_df):
    filters = [
        {"column": "quantity", "operator": "gt", "value": 1}
    ]

    result = apply_filters(sample_df, filters)

    assert result["quantity"].tolist() == [2, 3, 2]


def test_filter_greater_than_or_equal(sample_df):
    filters = [
        {"column": "quantity", "operator": "gte", "value": 2}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 3


def test_filter_less_than(sample_df):
    filters = [
        {"column": "quantity", "operator": "lt", "value": 2}
    ]

    result = apply_filters(sample_df, filters)

    assert result["quantity"].tolist() == [1, 1]


def test_filter_less_than_or_equal(sample_df):
    filters = [
        {"column": "quantity", "operator": "lte", "value": 1}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 2


def test_filter_contains(sample_df):
    filters = [
        {"column": "product", "operator": "contains", "value": "lap"}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 2
    assert all(result["product"] == "Laptop")


def test_filter_in(sample_df):
    filters = [
        {
            "column": "product",
            "operator": "in",
            "value": ["Laptop", "Mouse"],
        }
    ]

    result = apply_filters(sample_df, filters)

    assert set(result["product"]) == {"Laptop", "Mouse"}


def test_filter_not_in(sample_df):
    filters = [
        {
            "column": "product",
            "operator": "not_in",
            "value": ["Laptop", "Mouse"],
        }
    ]

    result = apply_filters(sample_df, filters)

    assert set(result["product"]) == {"Keyboard", "Monitor"}


def test_filter_is_null(sample_df):
    filters = [
        {"column": "age", "operator": "is_null"}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 1
    assert result["age"].isna().all()


def test_filter_not_null(sample_df):
    filters = [
        {"column": "age", "operator": "not_null"}
    ]

    result = apply_filters(sample_df, filters)

    assert len(result) == 4
    assert result["age"].notna().all()


def test_multiple_conditions_with_and(sample_df):
    filters = [
        {"column": "category", "operator": "eq", "value": "Electronics"},
        {"column": "quantity", "operator": "gt", "value": 1},
    ]

    result = apply_filters(sample_df, filters, "and")

    assert len(result) == 2
    assert all(result["category"] == "Electronics")
    assert all(result["quantity"] > 1)


def test_multiple_conditions_with_or(sample_df):
    filters = [
        {"column": "product", "operator": "eq", "value": "Laptop"},
        {"column": "product", "operator": "eq", "value": "Mouse"},
    ]

    result = apply_filters(sample_df, filters, "or")

    assert len(result) == 3
    assert set(result["product"]) == {"Laptop", "Mouse"}


def test_empty_filters_return_copy(sample_df):
    result = apply_filters(sample_df, [])

    pd.testing.assert_frame_equal(result, sample_df)
    assert result is not sample_df


def test_original_dataframe_is_not_modified(sample_df):
    original = sample_df.copy()

    filters = [
        {"column": "quantity", "operator": "gt", "value": 1}
    ]

    apply_filters(sample_df, filters)

    pd.testing.assert_frame_equal(sample_df, original)


def test_unknown_column_is_rejected(sample_df):
    filters = [
        {"column": "salary", "operator": "gt", "value": 1000}
    ]

    with pytest.raises(ValueError, match="Unknown filter column"):
        apply_filters(sample_df, filters)


def test_unknown_operator_is_rejected(sample_df):
    filters = [
        {"column": "quantity", "operator": "execute", "value": 1}
    ]

    with pytest.raises(ValueError, match="Unsupported filter operator"):
        apply_filters(sample_df, filters)


def test_invalid_logical_operator_is_rejected(sample_df):
    filters = [
        {"column": "quantity", "operator": "gt", "value": 1}
    ]

    with pytest.raises(ValueError, match="Logical operator"):
        apply_filters(sample_df, filters, "xor")


def test_non_numeric_comparison_is_rejected(sample_df):
    filters = [
        {"column": "product", "operator": "gt", "value": 100}
    ]

    with pytest.raises(ValueError, match="requires a numeric column"):
        apply_filters(sample_df, filters)


def test_missing_filter_value_is_rejected(sample_df):
    filters = [
        {"column": "quantity", "operator": "gt"}
    ]

    with pytest.raises(ValueError, match="requires a value"):
        apply_filters(sample_df, filters)


def test_invalid_in_value_is_rejected(sample_df):
    filters = [
        {"column": "product", "operator": "in", "value": "Laptop"}
    ]

    with pytest.raises(ValueError, match="requires a list"):
        apply_filters(sample_df, filters)


def test_empty_in_list_is_rejected(sample_df):
    filters = [
        {"column": "product", "operator": "in", "value": []}
    ]

    with pytest.raises(ValueError, match="non-empty list"):
        apply_filters(sample_df, filters)


def test_contains_requires_string(sample_df):
    filters = [
        {"column": "product", "operator": "contains", "value": 123}
    ]

    with pytest.raises(ValueError, match="requires a string"):
        apply_filters(sample_df, filters)


def test_validate_filters_accepts_valid_conditions(sample_df):
    filters = [
        {"column": "quantity", "operator": "gt", "value": 1}
    ]

    assert validate_filters(sample_df, filters) is True
