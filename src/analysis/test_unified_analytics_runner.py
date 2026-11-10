
import pandas as pd
import pytest

import unified_analytics_runner as runner


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": ["Laptop", "Mouse", "Mouse", "Keyboard"],
        "category": [
            "Electronics",
            "Accessories",
            "Accessories",
            "Accessories",
        ],
        "quantity": [1, 2, 3, 2],
        "unit_price": [55000, 400, 400, 1500],
    })


@pytest.fixture
def sample_plan():
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


def test_answer_unified_question(
    sample_df,
    sample_plan,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "interpret_unified_question",
        lambda question, df: sample_plan,
    )

    monkeypatch.setattr(
        runner,
        "explain_result",
        lambda question, plan, result: (
            "Accessories generated a total revenue of 5000."
        ),
    )

    response = runner.answer_unified_question(
        "Find Accessories revenue.",
        sample_df,
    )

    assert response["question"] == "Find Accessories revenue."
    assert response["unified_plan"] == sample_plan
    assert response["execution_result"]["filtered_row_count"] == 3

    assert response["query_result"]["result"] == [
        {"product": "Keyboard", "revenue": 3000},
        {"product": "Mouse", "revenue": 2000},
    ]

    assert "5000" in response["explanation"]


def test_empty_question_rejected(sample_df):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        runner.answer_unified_question("", sample_df)


def test_invalid_question_type_rejected(sample_df):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        runner.answer_unified_question(None, sample_df)


def test_interpreter_error_propagated(sample_df, monkeypatch):

    def failing_interpreter(question, df):
        raise RuntimeError("Interpreter failed.")

    monkeypatch.setattr(
        runner,
        "interpret_unified_question",
        failing_interpreter,
    )

    with pytest.raises(RuntimeError, match="Interpreter failed"):
        runner.answer_unified_question("Find revenue.", sample_df)


def test_explanation_error_propagated(
    sample_df,
    sample_plan,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "interpret_unified_question",
        lambda question, df: sample_plan,
    )

    def failing_explainer(question, plan, result):
        raise RuntimeError("Explanation failed.")

    monkeypatch.setattr(
        runner,
        "explain_result",
        failing_explainer,
    )

    with pytest.raises(RuntimeError, match="Explanation failed"):
        runner.answer_unified_question("Find revenue.", sample_df)


def test_report_displays_result(
    sample_df,
    sample_plan,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        runner,
        "interpret_unified_question",
        lambda question, df: sample_plan,
    )

    monkeypatch.setattr(
        runner,
        "explain_result",
        lambda question, plan, result: "Revenue analysis complete.",
    )

    response = runner.answer_unified_question(
        "Find Accessories revenue.",
        sample_df,
    )

    runner.print_unified_report(response)

    output = capsys.readouterr().out

    assert "DATASLEUTH UNIFIED ANALYTICS" in output
    assert "Accessories" in output
    assert "Calculated result" in output
    assert "Revenue analysis complete." in output


def test_unsupported_currency_symbol_removed(
    sample_df,
    sample_plan,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "interpret_unified_question",
        lambda question, df: sample_plan,
    )

    monkeypatch.setattr(
        runner,
        "explain_result",
        lambda question, plan, result: (
            "Laptop generated $165,000 in revenue."
        ),
    )

    response = runner.answer_unified_question(
        "Find highest revenue.",
        sample_df,
    )

    assert "$" not in response["explanation"]
    assert "165,000" in response["explanation"]


def test_currency_symbol_preserved_with_currency_metadata():
    df = pd.DataFrame({
        "product": ["Laptop"],
        "revenue": [165000],
        "currency": ["USD"],
    })

    explanation = "Laptop generated $165,000 in revenue."

    result = runner.normalize_explanation_units(
        explanation,
        df,
    )

    assert result == explanation


def test_multiple_unsupported_currency_symbols_removed():
    explanation = (
        "Revenue values: $100, ₹200, €300, £400 and ¥500."
    )

    result = runner.normalize_explanation_units(
        explanation,
        pd.DataFrame({"revenue": [100, 200, 300]}),
    )

    assert "$" not in result
    assert "₹" not in result
    assert "€" not in result
    assert "£" not in result
    assert "¥" not in result

    assert "100" in result
    assert "200" in result
    assert "300" in result
    assert "400" in result
    assert "500" in result


def test_currency_code_in_column_name_preserves_symbol():
    df = pd.DataFrame({
        "product": ["Laptop"],
        "revenue_usd": [165000],
    })

    explanation = "Laptop generated $165,000."

    result = runner.normalize_explanation_units(
        explanation,
        df,
    )

    assert result == explanation


def test_non_string_explanation_rejected(sample_df):
    with pytest.raises(ValueError, match="Explanation must be a string"):
        runner.normalize_explanation_units(12345, sample_df)
