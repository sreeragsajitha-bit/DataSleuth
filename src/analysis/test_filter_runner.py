
import pandas as pd
import pytest

import filter_runner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard", "Laptop", "Monitor"],
        "category": [
            "Electronics",
            "Accessories",
            "Accessories",
            "Electronics",
            "Electronics",
        ],
        "quantity": [1, 2, 1, 3, 2],
        "unit_price": [55000, 400, 1500, 55000, 12000],
    })


@pytest.fixture
def sample_plan():
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


def test_answer_filter_question_returns_matching_rows(
    sample_df, sample_plan, monkeypatch
):
    monkeypatch.setattr(
        filter_runner,
        "interpret_filter_question",
        lambda question, df: sample_plan,
    )

    response = filter_runner.answer_filter_question(
        "Show electronics products where quantity is greater than 1.",
        sample_df,
    )

    assert response["row_count"] == 2
    assert set(response["result"]["product"]) == {"Laptop", "Monitor"}


def test_answer_filter_question_returns_metadata(
    sample_df, sample_plan, monkeypatch
):
    monkeypatch.setattr(
        filter_runner,
        "interpret_filter_question",
        lambda question, df: sample_plan,
    )

    response = filter_runner.answer_filter_question(
        "Find matching products",
        sample_df,
    )

    assert response["original_row_count"] == 5
    assert response["row_count"] == 2
    assert response["filter_plan"] == sample_plan


def test_answer_filter_question_rejects_empty_question(sample_df):
    with pytest.raises(ValueError, match="cannot be empty"):
        filter_runner.answer_filter_question("", sample_df)


def test_answer_filter_question_rejects_whitespace(sample_df):
    with pytest.raises(ValueError, match="cannot be empty"):
        filter_runner.answer_filter_question("   ", sample_df)


def test_answer_filter_question_propagates_interpreter_error(
    sample_df, monkeypatch
):
    def failing_interpreter(question, df):
        raise ValueError("Unsupported filter question")

    monkeypatch.setattr(
        filter_runner,
        "interpret_filter_question",
        failing_interpreter,
    )

    with pytest.raises(ValueError, match="Unsupported filter question"):
        filter_runner.answer_filter_question("Find products", sample_df)


def test_answer_filter_question_does_not_modify_original(
    sample_df, sample_plan, monkeypatch
):
    original = sample_df.copy()

    monkeypatch.setattr(
        filter_runner,
        "interpret_filter_question",
        lambda question, df: sample_plan,
    )

    filter_runner.answer_filter_question("Find products", sample_df)

    pd.testing.assert_frame_equal(sample_df, original)


def test_print_filter_report(
    sample_df, sample_plan, monkeypatch, capsys
):
    monkeypatch.setattr(
        filter_runner,
        "interpret_filter_question",
        lambda question, df: sample_plan,
    )

    response = filter_runner.answer_filter_question(
        "Show electronics products where quantity is greater than 1.",
        sample_df,
    )

    filter_runner.print_filter_report(response)

    output = capsys.readouterr().out

    assert "DATASLEUTH INTELLIGENT FILTER" in output
    assert "Laptop" in output
    assert "Monitor" in output
    assert "Records found: 2" in output
