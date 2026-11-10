
import json

import pandas as pd
import pytest

from unified_interpreter import (
    build_unified_prompt,
    parse_unified_plan,
)
from unified_analytics import validate_unified_plan


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard"],
        "category": [
            "Electronics",
            "Accessories",
            "Accessories",
        ],
        "quantity": [1, 2, 3],
        "unit_price": [55000, 400, 1500],
    })


@pytest.fixture
def valid_plan():
    return {
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
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 2,
    }


def test_parse_valid_unified_plan(valid_plan):
    response = json.dumps(valid_plan)

    result = parse_unified_plan(response)

    assert result == valid_plan


def test_parse_json_inside_code_fence(valid_plan):
    response = (
        "```json\n"
        + json.dumps(valid_plan)
        + "\n```"
    )

    result = parse_unified_plan(response)

    assert result == valid_plan


def test_invalid_json_rejected():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_unified_plan("{invalid json}")


def test_empty_plan_rejected():
    with pytest.raises(ValueError, match="supported unified plan"):
        parse_unified_plan("{}")


def test_non_dictionary_plan_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        parse_unified_plan('["not", "a", "dictionary"]')


def test_valid_plan_passes_validation(sample_df, valid_plan):
    assert validate_unified_plan(sample_df, valid_plan) is True


def test_invalid_filter_column_rejected(sample_df, valid_plan):
    valid_plan["filters"][0]["column"] = "unknown_column"

    with pytest.raises(ValueError, match="Unknown filter column"):
        validate_unified_plan(sample_df, valid_plan)


def test_invalid_aggregation_rejected(sample_df, valid_plan):
    valid_plan["aggregation"] = "calculate_everything"

    with pytest.raises(ValueError, match="Unsupported aggregation"):
        validate_unified_plan(sample_df, valid_plan)


def test_unsupported_plan_field_rejected(sample_df, valid_plan):
    valid_plan["python_code"] = "print('hello')"

    with pytest.raises(ValueError, match="Unsupported unified plan fields"):
        validate_unified_plan(sample_df, valid_plan)


def test_prompt_contains_dataset_schema(sample_df):
    prompt = build_unified_prompt(
        "Find total revenue for Accessories",
        sample_df,
    )

    assert "category" in prompt
    assert "Accessories" in prompt
    assert "quantity" in prompt
    assert "ALLOWED FILTER OPERATORS" in prompt
    assert "ALLOWED AGGREGATIONS" in prompt


def test_prompt_contains_user_question(sample_df):
    question = "Find the highest revenue product"

    prompt = build_unified_prompt(question, sample_df)

    assert question in prompt


def test_prompt_does_not_request_executable_code(sample_df):
    prompt = build_unified_prompt("Find total revenue", sample_df)

    assert "Never generate Python code or SQL." in prompt
