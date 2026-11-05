
import pandas as pd
import pytest

import filtered_aggregation


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": [
            "Laptop",
            "Mouse",
            "Keyboard",
            "Mouse",
            "Headphones",
            "Monitor",
        ],
        "category": [
            "Electronics",
            "Accessories",
            "Accessories",
            "Accessories",
            "Accessories",
            "Electronics",
        ],
        "quantity": [1, 2, 2, 3, 2, 1],
        "unit_price": [55000, 400, 1500, 400, 2000, 12000],
    })


@pytest.fixture
def accessories_plan():
    return {
        "filters": [
            {
                "column": "category",
                "operator": "eq",
                "value": "accessories",
            },
            {
                "column": "quantity",
                "operator": "gt",
                "value": 1,
            },
        ],
        "logical_operator": "and",
    }


def test_filtered_revenue_sum(sample_df, accessories_plan, monkeypatch):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    response = filtered_aggregation.answer_filtered_aggregation(
        "Calculate accessories revenue where quantity is greater than 1.",
        sample_df,
    )

    assert response["aggregation_result"]["result"] == 9000


def test_filtered_row_count(sample_df, accessories_plan, monkeypatch):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    response = filtered_aggregation.answer_filtered_aggregation(
        "Find matching accessories.",
        sample_df,
    )

    assert response["filtered_row_count"] == 4


def test_filtered_data_contains_matching_products(
    sample_df, accessories_plan, monkeypatch
):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    response = filtered_aggregation.answer_filtered_aggregation(
        "Find matching accessories.",
        sample_df,
    )

    assert set(response["filtered_data"]["product"]) == {
        "Mouse",
        "Keyboard",
        "Headphones",
    }


def test_original_dataframe_is_not_modified(
    sample_df, accessories_plan, monkeypatch
):
    original = sample_df.copy()

    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    filtered_aggregation.answer_filtered_aggregation(
        "Find matching accessories.",
        sample_df,
    )

    pd.testing.assert_frame_equal(sample_df, original)


def test_empty_question_is_rejected(sample_df):
    with pytest.raises(ValueError, match="cannot be empty"):
        filtered_aggregation.answer_filtered_aggregation("", sample_df)


def test_invalid_metric_is_rejected(
    sample_df, accessories_plan, monkeypatch
):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    with pytest.raises(ValueError, match="Unknown metric"):
        filtered_aggregation.answer_filtered_aggregation(
            "Find matching accessories.",
            sample_df,
            metric="salary",
        )


def test_mean_aggregation(sample_df, accessories_plan, monkeypatch):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    response = filtered_aggregation.answer_filtered_aggregation(
        "Find matching accessories.",
        sample_df,
        metric="quantity",
        aggregation="mean",
    )

    assert response["aggregation_result"]["result"] == 2.25


def test_grouped_aggregation(
    sample_df, accessories_plan, monkeypatch
):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    response = filtered_aggregation.answer_filtered_aggregation(
        "Find matching accessories.",
        sample_df,
        metric="revenue",
        aggregation="sum",
        group_by="product",
    )

    results = {
        row["product"]: row["revenue"]
        for row in response["aggregation_result"]["result"]
    }

    assert results == {
        "Mouse": 2000,
        "Keyboard": 3000,
        "Headphones": 4000,
    }


def test_interpreter_error_is_propagated(sample_df, monkeypatch):
    def failing_interpreter(question, df):
        raise ValueError("Invalid filter question")

    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        failing_interpreter,
    )

    with pytest.raises(ValueError, match="Invalid filter question"):
        filtered_aggregation.answer_filtered_aggregation(
            "Find matching accessories.",
            sample_df,
        )


def test_report_displays_result(
    sample_df, accessories_plan, monkeypatch, capsys
):
    monkeypatch.setattr(
        filtered_aggregation,
        "interpret_filter_question",
        lambda question, df: accessories_plan,
    )

    response = filtered_aggregation.answer_filtered_aggregation(
        "Calculate accessories revenue.",
        sample_df,
    )

    filtered_aggregation.print_aggregation_report(response)

    output = capsys.readouterr().out

    assert "DATASLEUTH FILTERED AGGREGATION" in output
    assert "9000" in output
