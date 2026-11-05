
import json

import pandas as pd
import pytest

from query_interpreter import (
    build_prompt,
    extract_response_content,
    parse_query_plan,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard"],
        "quantity": [1, 2, 1],
        "unit_price": [55000, 400, 1500],
    })


def test_build_prompt_includes_question(sample_df):
    prompt = build_prompt(
        "Which product generated the highest revenue?",
        sample_df,
    )

    assert "Which product generated the highest revenue?" in prompt


def test_build_prompt_includes_dataset_columns(sample_df):
    prompt = build_prompt("Calculate total revenue", sample_df)

    assert "product" in prompt
    assert "quantity" in prompt
    assert "unit_price" in prompt
    assert "revenue" in prompt


def test_build_prompt_disallows_python_and_sql(sample_df):
    prompt = build_prompt("Calculate total revenue", sample_df)

    assert "Never generate Python code or SQL." in prompt


def test_parse_valid_query_plan():
    response = """
    {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 1
    }
    """

    plan = parse_query_plan(response)

    assert plan["metric"] == "revenue"
    assert plan["aggregation"] == "sum"
    assert plan["group_by"] == "product"
    assert plan["limit"] == 1


def test_parse_json_inside_markdown_fence():
    response = """```json
    {
        "operation": "aggregate",
        "metric": "quantity",
        "aggregation": "mean"
    }
    ```"""

    plan = parse_query_plan(response)

    assert plan["metric"] == "quantity"
    assert plan["aggregation"] == "mean"


def test_invalid_json_is_rejected():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_query_plan("{invalid json}")


def test_empty_plan_is_rejected():
    with pytest.raises(ValueError, match="could not create"):
        parse_query_plan("{}")


def test_non_dictionary_json_is_rejected():
    with pytest.raises(ValueError, match="JSON object"):
        parse_query_plan('["sum", "mean"]')


def test_empty_response_is_rejected():
    with pytest.raises(ValueError, match="must be a string"):
        parse_query_plan(None)


def test_extract_content_from_dictionary():
    response = {
        "message": {
            "content": '{"operation": "aggregate"}'
        }
    }

    content = extract_response_content(response)

    assert json.loads(content)["operation"] == "aggregate"


def test_extract_content_from_object():
    class Message:
        content = '{"metric": "quantity"}'

    class Response:
        message = Message()

    content = extract_response_content(Response())

    assert json.loads(content)["metric"] == "quantity"


def test_empty_model_content_is_rejected():
    response = {
        "message": {
            "content": ""
        }
    }

    with pytest.raises(ValueError, match="empty response"):
        extract_response_content(response)


def test_prompt_includes_allowed_aggregations(sample_df):
    prompt = build_prompt("Calculate something", sample_df)

    for aggregation in ["sum", "mean", "median", "min", "max", "count"]:
        assert aggregation in prompt
