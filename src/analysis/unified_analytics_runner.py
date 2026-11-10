
import os
import re
import sys

from unified_interpreter import interpret_unified_question
from unified_analytics import execute_unified_plan
from result_explainer import explain_result


CURRENCY_SYMBOL_PATTERN = re.compile(
    r"(?<!\w)[$₹€£¥]\s*(?=\d)"
)

CURRENCY_METADATA_MARKERS = {
    "currency",
    "currency_code",
    "currency_symbol",
    "monetary_unit",
    "currency_unit",
}

CURRENCY_CODES = {
    "usd", "inr", "eur", "gbp", "jpy",
    "aud", "cad", "sgd", "dollar", "dollars",
}


def normalize_explanation_units(explanation, df):
    """
    Remove unsupported currency symbols from AI explanations.

    Currency symbols are preserved when the dataset contains
    explicit currency metadata in its column names.
    """

    if not isinstance(explanation, str):
        raise ValueError("Explanation must be a string.")

    column_names = {
        str(column).strip().lower()
        for column in df.columns
    }

    has_currency_metadata = any(
        marker in column_names
        for marker in CURRENCY_METADATA_MARKERS
    )

    if not has_currency_metadata:
        has_currency_metadata = any(
            any(
                marker in column.split("_")
                for marker in CURRENCY_CODES
            )
            for column in column_names
        )

    if has_currency_metadata:
        return explanation

    return CURRENCY_SYMBOL_PATTERN.sub("", explanation)


def answer_unified_question(question, df):
    """
    Complete natural-language analytics workflow.

    1. Interpret the user's question using Qwen.
    2. Validate the unified analytics plan.
    3. Execute filtering and aggregation using Python.
    4. Explain the calculated result using Qwen.
    5. Remove unsupported currency symbols.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Interpret the question using the unified AI interpreter.
    unified_plan = interpret_unified_question(
        question,
        df,
    )

    # Step 2: Execute the validated plan using Python.
    execution_result = execute_unified_plan(
        df,
        unified_plan,
    )

    # Step 3: Explain the actual calculated result.
    explanation = explain_result(
        question,
        unified_plan,
        execution_result["query_result"],
    )

    # Step 4: Enforce dataset-supported units.
    explanation = normalize_explanation_units(
        explanation,
        df,
    )

    return {
        "question": question,
        "unified_plan": unified_plan,
        "execution_result": execution_result,
        "query_result": execution_result["query_result"],
        "explanation": explanation,
    }


def print_unified_report(response):
    """Display the unified analytics report."""

    plan = response["unified_plan"]
    execution = response["execution_result"]
    query_result = response["query_result"]

    print("\n" + "=" * 60)
    print("DATASLEUTH UNIFIED ANALYTICS")
    print("=" * 60)

    print("\nQuestion:")
    print(response["question"])

    print("\nApplied filters:")

    filters = plan.get("filters", [])

    if filters:
        for condition in filters:
            print(
                f"{condition['column']} "
                f"{condition['operator']} "
                f"{condition['value']}"
            )
    else:
        print("No filters applied.")

    print(
        "\nLogical operator:",
        plan.get("logical_operator", "and").upper(),
    )

    print("\nAggregation:")
    print("Metric:", plan.get("metric"))
    print("Operation:", plan.get("aggregation"))
    print("Grouped by:", plan.get("group_by"))

    print("Sorted by:", plan.get("sort_by"))
    print("Sort order:", plan.get("sort_order"))
    print("Limit:", plan.get("limit"))

    print("\nRecords analysed:")
    print(
        f"{execution['filtered_row_count']} "
        f"of {execution['original_row_count']}"
    )

    print("\nCalculated result:")

    for record in query_result["result"]:
        print(record)

    print("\nAI Explanation:")
    print(response["explanation"])

    print("\n" + "=" * 60)


if __name__ == "__main__":

    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    import pandas as pd

    DATASET_PATH = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "sales.csv",
    )

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    question = (
        "Which product generated the highest revenue?"
    )

    response = answer_unified_question(
        question,
        df,
    )

    print_unified_report(response)
