
import pytest

from result_explainer import (
    build_explanation_prompt,
    extract_explanation,
    explain_result,
)


@pytest.fixture
def query_data():
    question = "Which product generated the highest revenue?"

    plan = {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 1,
    }

    result = {
        "success": True,
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "result": [
            {"product": "Laptop", "revenue": 165000}
        ],
        "row_count": 1,
    }

    return question, plan, result


def test_prompt_contains_original_question(query_data):
    question, plan, result = query_data

    prompt = build_explanation_prompt(question, plan, result)

    assert question in prompt


def test_prompt_contains_calculated_result(query_data):
    question, plan, result = query_data

    prompt = build_explanation_prompt(question, plan, result)

    assert "165000" in prompt
    assert "Laptop" in prompt


def test_prompt_prevents_invented_statistics(query_data):
    question, plan, result = query_data

    prompt = build_explanation_prompt(question, plan, result)

    assert "Never invent statistics" in prompt
    assert "Never modify or recalculate" in prompt


def test_extract_explanation_from_dictionary():
    response = {
        "message": {
            "content": "Laptop generated the highest revenue."
        }
    }

    explanation = extract_explanation(response)

    assert explanation == "Laptop generated the highest revenue."


def test_extract_explanation_from_object():
    class Message:
        content = "Laptop generated the highest revenue."

    class Response:
        message = Message()

    explanation = extract_explanation(Response())

    assert explanation == "Laptop generated the highest revenue."


def test_empty_explanation_is_rejected():
    response = {
        "message": {
            "content": ""
        }
    }

    with pytest.raises(ValueError, match="empty explanation"):
        extract_explanation(response)


def test_non_string_explanation_is_rejected():
    with pytest.raises(ValueError):
        extract_explanation(None)


def test_empty_question_is_rejected(query_data):
    _, plan, result = query_data

    with pytest.raises(ValueError, match="cannot be empty"):
        explain_result("", plan, result)


def test_invalid_query_plan_is_rejected(query_data):
    question, _, result = query_data

    with pytest.raises(ValueError, match="Query plan"):
        explain_result(question, [], result)


def test_invalid_query_result_is_rejected(query_data):
    question, plan, _ = query_data

    with pytest.raises(ValueError, match="Query result"):
        explain_result(question, plan, [])


def test_unsuccessful_query_is_rejected(query_data):
    question, plan, _ = query_data

    result = {"success": False}

    with pytest.raises(ValueError, match="unsuccessful query"):
        explain_result(question, plan, result)
