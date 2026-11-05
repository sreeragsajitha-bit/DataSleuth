
import pandas as pd
import pytest

from filter_interpreter import (
    build_filter_prompt,
    parse_filter_plan,
    validate_filter_plan,
    extract_response_content,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard"],
        "category": ["Electronics", "Accessories", "Accessories"],
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
                "value": "Electronics",
            },
            {
                "column": "quantity",
                "operator": "gt",
                "value": 1,
            },
        ],
        "logical_operator": "and",
    }


def test_prompt_includes_question(sample_df):
    question = "Show electronics products."

    prompt = build_filter_prompt(question, sample_df)

    assert question in prompt


def test_prompt_includes_dataset_columns(sample_df):
    prompt = build_filter_prompt("Filter products", sample_df)

    assert "product" in prompt
    assert "category" in prompt
    assert "quantity" in prompt
    assert "revenue" in prompt


def test_prompt_includes_allowed_operators(sample_df):
    prompt = build_filter_prompt("Filter products", sample_df)

    assert "eq: equal to" in prompt
    assert "gt: greater than" in prompt
    assert "contains: text contains" in prompt
    assert "is_null: value is missing" in prompt


def test_prompt_disallows_code_generation(sample_df):
    prompt = build_filter_prompt("Filter products", sample_df)

    assert "Never generate Python code or SQL." in prompt


def test_parse_valid_filter_plan(valid_plan):
    import json

    plan = parse_filter_plan(json.dumps(valid_plan))

    assert plan["logical_operator"] == "and"
    assert len(plan["filters"]) == 2


def test_parse_markdown_json():
    response = """```json
    {
        "filters": [
            {
                "column": "quantity",
                "operator": "gt",
                "value": 2
            }
        ],
        "logical_operator": "and"
    }
    ```"""

    plan = parse_filter_plan(response)

    assert plan["filters"][0]["value"] == 2


def test_invalid_json_is_rejected():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_filter_plan("{invalid}")


def test_empty_plan_is_rejected():
    with pytest.raises(ValueError, match="could not create"):
        parse_filter_plan("{}")


def test_non_dictionary_json_is_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        parse_filter_plan('["quantity", "gt", 2]')


def test_non_string_response_is_rejected():
    with pytest.raises(ValueError, match="must be a string"):
        parse_filter_plan(None)


def test_valid_filter_plan_is_accepted(sample_df, valid_plan):
    assert validate_filter_plan(sample_df, valid_plan) is True


def test_unknown_plan_field_is_rejected(sample_df):
    plan = {
        "filters": [
            {
                "column": "quantity",
                "operator": "gt",
                "value": 1,
            }
        ],
        "python_code": "import os",
    }

    with pytest.raises(ValueError, match="Unsupported filter plan fields"):
        validate_filter_plan(sample_df, plan)


def test_empty_filters_are_rejected(sample_df):
    plan = {
        "filters": [],
        "logical_operator": "and",
    }

    with pytest.raises(ValueError, match="at least one condition"):
        validate_filter_plan(sample_df, plan)


def test_invalid_column_is_rejected(sample_df):
    plan = {
        "filters": [
            {
                "column": "salary",
                "operator": "gt",
                "value": 100,
            }
        ]
    }

    with pytest.raises(ValueError, match="Unknown filter column"):
        validate_filter_plan(sample_df, plan)


def test_invalid_operator_is_rejected(sample_df):
    plan = {
        "filters": [
            {
                "column": "quantity",
                "operator": "execute",
                "value": 1,
            }
        ]
    }

    with pytest.raises(ValueError, match="Unsupported filter operator"):
        validate_filter_plan(sample_df, plan)


def test_invalid_logical_operator_is_rejected(sample_df):
    plan = {
        "filters": [
            {
                "column": "quantity",
                "operator": "gt",
                "value": 1,
            }
        ],
        "logical_operator": "xor",
    }

    with pytest.raises(ValueError, match="Logical operator"):
        validate_filter_plan(sample_df, plan)


def test_extract_response_from_dictionary():
    response = {
        "message": {
            "content": '{"filters": []}'
        }
    }

    assert extract_response_content(response) == '{"filters": []}'


def test_extract_response_from_object():
    class Message:
        content = '{"filters": []}'

    class Response:
        message = Message()

    assert extract_response_content(Response()) == '{"filters": []}'


def test_empty_model_response_is_rejected():
    response = {
        "message": {
            "content": ""
        }
    }

    with pytest.raises(ValueError, match="empty response"):
        extract_response_content(response)
