
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


ALLOWED_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "in",
    "not_in",
    "is_null",
    "not_null",
}

ALLOWED_LOGICAL_OPERATORS = {"and", "or"}

OPERATORS_REQUIRING_VALUE = ALLOWED_OPERATORS - {
    "is_null",
    "not_null",
}


def is_text_column(series):
    """Check whether a column can be treated as text."""
    return (
        is_string_dtype(series.dtype)
        or is_object_dtype(series.dtype)
    )


def normalize_text(series):
    """Normalize text for case-insensitive comparisons."""
    return series.astype("string").str.strip().str.casefold()


def validate_filters(df, filters, logical_operator="and"):
    """Validate filtering conditions before applying them."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if not isinstance(filters, list):
        raise ValueError("Filters must be provided as a list.")

    if logical_operator not in ALLOWED_LOGICAL_OPERATORS:
        raise ValueError("Logical operator must be 'and' or 'or'.")

    for condition in filters:
        if not isinstance(condition, dict):
            raise ValueError("Each filter must be a dictionary.")

        allowed_keys = {"column", "operator", "value"}

        if set(condition.keys()) - allowed_keys:
            raise ValueError("Filter contains unsupported fields.")

        column = condition.get("column")
        operator = condition.get("operator")

        if not isinstance(column, str):
            raise ValueError("Filter column must be a string.")

        if column not in df.columns:
            raise ValueError(f"Unknown filter column: {column}")

        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {operator}")

        if operator in OPERATORS_REQUIRING_VALUE:
            if "value" not in condition:
                raise ValueError(
                    f"Operator '{operator}' requires a value."
                )

        if operator in {"in", "not_in"}:
            if not isinstance(condition["value"], list):
                raise ValueError(
                    f"Operator '{operator}' requires a list of values."
                )

            if not condition["value"]:
                raise ValueError(
                    f"Operator '{operator}' requires a non-empty list."
                )

        if operator == "contains":
            if not isinstance(condition.get("value"), str):
                raise ValueError(
                    "The contains operator requires a string value."
                )

        if operator in {"gt", "gte", "lt", "lte"}:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(
                    f"Comparison operator '{operator}' requires "
                    f"a numeric column: {column}"
                )

            value = condition["value"]

            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError(
                    f"Comparison value for '{operator}' must be numeric."
                )

    return True


def apply_filters(df, filters, logical_operator="and"):
    """
    Apply validated filtering conditions.

    Text equality and membership comparisons are case-insensitive.
    Numeric comparisons retain their normal behavior.

    The original DataFrame is never modified.
    """

    validate_filters(df, filters, logical_operator)

    if not filters:
        return df.copy()

    masks = []

    for condition in filters:
        column = condition["column"]
        operator = condition["operator"]
        value = condition.get("value")
        series = df[column]

        if operator == "eq":
            if isinstance(value, str) and is_text_column(series):
                mask = normalize_text(series) == value.strip().casefold()
            else:
                mask = series == value

        elif operator == "ne":
            if isinstance(value, str) and is_text_column(series):
                mask = normalize_text(series) != value.strip().casefold()
            else:
                mask = series != value

        elif operator == "gt":
            mask = series > value

        elif operator == "gte":
            mask = series >= value

        elif operator == "lt":
            mask = series < value

        elif operator == "lte":
            mask = series <= value

        elif operator == "contains":
            mask = series.astype("string").str.contains(
                value,
                case=False,
                regex=False,
                na=False,
            )

        elif operator == "in":
            if (
                is_text_column(series)
                and all(isinstance(item, str) for item in value)
            ):
                normalized_values = [
                    item.strip().casefold() for item in value
                ]
                mask = normalize_text(series).isin(normalized_values)
            else:
                mask = series.isin(value)

        elif operator == "not_in":
            if (
                is_text_column(series)
                and all(isinstance(item, str) for item in value)
            ):
                normalized_values = [
                    item.strip().casefold() for item in value
                ]
                mask = ~normalize_text(series).isin(normalized_values)
            else:
                mask = ~series.isin(value)

        elif operator == "is_null":
            mask = series.isna()

        elif operator == "not_null":
            mask = series.notna()

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        masks.append(mask.fillna(False))

    combined_mask = masks[0]

    for mask in masks[1:]:
        if logical_operator == "and":
            combined_mask = combined_mask & mask
        else:
            combined_mask = combined_mask | mask

    return df.loc[combined_mask].copy()
