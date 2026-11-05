
from filter_interpreter import interpret_filter_question
from filter_engine import apply_filters
from query_engine import execute_query


def answer_filtered_aggregation(
    question,
    df,
    metric="revenue",
    aggregation="sum",
    group_by=None,
):
    """
    Filter a dataset using Qwen and aggregate the matching rows.

    Qwen determines the filter conditions.
    Python validates and applies those conditions.
    Python performs the final aggregation.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Ask Qwen to interpret the filtering conditions.
    filter_plan = interpret_filter_question(question, df)

    # Step 2: Apply the validated filters using Python.
    filtered_df = apply_filters(
        df,
        filter_plan["filters"],
        filter_plan.get("logical_operator", "and"),
    )

    # Step 3: Build a controlled aggregation plan.
    aggregation_plan = {
        "operation": "aggregate",
        "metric": metric,
        "aggregation": aggregation,
    }

    if group_by is not None:
        aggregation_plan["group_by"] = group_by

    # Step 4: Execute the aggregation using the query engine.
    aggregation_result = execute_query(
        filtered_df,
        aggregation_plan,
    )

    return {
        "question": question,
        "filter_plan": filter_plan,
        "filtered_row_count": len(filtered_df),
        "filtered_data": filtered_df,
        "aggregation_plan": aggregation_plan,
        "aggregation_result": aggregation_result,
    }


def print_aggregation_report(response):
    """Display filtering and aggregation results."""

    print("\n" + "=" * 55)
    print("DATASLEUTH FILTERED AGGREGATION")
    print("=" * 55)

    print("\nQuestion:")
    print(response["question"])

    print("\nApplied filters:")

    for condition in response["filter_plan"]["filters"]:
        print(
            f"{condition['column']} "
            f"{condition['operator']} "
            f"{condition.get('value', '(no value)')}"
        )

    print(
        "\nLogical operator:",
        response["filter_plan"].get("logical_operator", "and").upper(),
    )

    print("\nMatching records:")
    print(response["filtered_data"].to_string(index=False))

    print("\nMatching record count:", response["filtered_row_count"])

    print("\nAggregation:")
    print(response["aggregation_plan"])

    print("\nCalculated result:")
    print(response["aggregation_result"]["result"])

    print("\n" + "=" * 55)


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    question = (
        "What is the total revenue of accessories products "
        "where quantity is greater than 1?"
    )

    response = answer_filtered_aggregation(
        question,
        df,
        metric="revenue",
        aggregation="sum",
    )

    print_aggregation_report(response)
