
import pandas as pd
import pytest

import query_runner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard", "Laptop", "Mouse"],
        "quantity": [1, 2, 1, 2, 3],
        "unit_price": [55000, 400, 1500, 55000, 400],
    })


@pytest.fixture
def query_plan():
    return {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 1,
    }


def mock_interpreter(question, df):
    return {
        "operation": "aggregate",
        "metric": "revenue",
        "aggregation": "sum",
        "group_by": "product",
        "sort_by": "revenue",
        "sort_order": "desc",
        "limit": 1,
    }


def mock_explainer(question, plan, result):
    return "Laptop generated the highest revenue."


def test_answer_question_returns_complete_response(
    sample_df, monkeypatch
):
    monkeypatch.setattr(
        query_runner,
        "interpret_question",
        mock_interpreter,
    )

    monkeypatch.setattr(
        query_runner,
        "explain_result",
        mock_explainer,
    )

    response = query_runner.answer_question(
        "Which product generated the highest revenue?",
        sample_df,
    )

    assert response["query_result"]["result"][0]["product"] == "Laptop"
    assert response["query_result"]["result"][0]["revenue"] == 165000
    assert response["explanation"] == (
        "Laptop generated the highest revenue."
    )


def test_answer_question_rejects_empty_question(sample_df):
    with pytest.raises(ValueError, match="cannot be empty"):
        query_runner.answer_question("", sample_df)


def test_answer_question_rejects_whitespace_question(sample_df):
    with pytest.raises(ValueError, match="cannot be empty"):
        query_runner.answer_question("   ", sample_df)


def test_answer_question_propagates_interpreter_error(
    sample_df, monkeypatch
):
    def failing_interpreter(question, df):
        raise ValueError("Unsupported question")

    monkeypatch.setattr(
        query_runner,
        "interpret_question",
        failing_interpreter,
    )

    with pytest.raises(ValueError, match="Unsupported question"):
        query_runner.answer_question("Do something", sample_df)


def test_answer_question_does_not_modify_dataset(
    sample_df, monkeypatch
):
    original = sample_df.copy()

    monkeypatch.setattr(
        query_runner,
        "interpret_question",
        mock_interpreter,
    )

    monkeypatch.setattr(
        query_runner,
        "explain_result",
        mock_explainer,
    )

    query_runner.answer_question("Calculate revenue", sample_df)

    pd.testing.assert_frame_equal(sample_df, original)


def test_print_answer_report(sample_df, monkeypatch, capsys):
    monkeypatch.setattr(
        query_runner,
        "interpret_question",
        mock_interpreter,
    )

    monkeypatch.setattr(
        query_runner,
        "explain_result",
        mock_explainer,
    )

    response = query_runner.answer_question(
        "Which product generated the highest revenue?",
        sample_df,
    )

    query_runner.print_answer_report(response)

    output = capsys.readouterr().out

    assert "DATASLEUTH INTELLIGENT QUERY" in output
    assert "Laptop" in output
    assert "165000" in output
    assert "AI Explanation" in output


def test_answer_question_propagates_explainer_error(
    sample_df, monkeypatch
):
    monkeypatch.setattr(
        query_runner,
        "interpret_question",
        mock_interpreter,
    )

    def failing_explainer(question, plan, result):
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(
        query_runner,
        "explain_result",
        failing_explainer,
    )

    with pytest.raises(RuntimeError, match="Ollama unavailable"):
        query_runner.answer_question("Calculate revenue", sample_df)
