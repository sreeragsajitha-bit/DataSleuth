
from filter_interpreter import interpret_filter_question
from filter_engine import apply_filters


def answer_filter_question(question, df):
    """
    Complete natural-language filtering workflow.

    1. Interpret the user's question using Qwen.
    2. Apply validated filters using Python.
    3. Return matching records and metadata.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Generate a validated filter plan.
    filter_plan = interpret_filter_question(question, df)

    # Step 2: Extract filtering instructions.
    filters = filter_plan["filters"]
    logical_operator = filter_plan.get("logical_operator", "and")

    # Step 3: Apply filters using Python.
    filtered_df = apply_filters(
        df,
        filters,
        logical_operator,
    )

    return {
        "question": question,
        "filter_plan": filter_plan,
        "result": filtered_df,
        "row_count": len(filtered_df),
        "original_row_count": len(df),
    }


def print_filter_report(response):
    """
    Display the filtering question, conditions and matching records.
    """

    print("\n" + "=" * 55)
    print("DATASLEUTH INTELLIGENT FILTER")
    print("=" * 55)

    print("\nQuestion:")
    print(response["question"])

    print("\nFilter conditions:")

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
    print(response["result"].to_string(index=False))

    print("\nRecords found:", response["row_count"])
    print("Original records:", response["original_row_count"])

    print("\n" + "=" * 55)


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    question = (
    "Show accessories products where quantity is greater than 1."
)

    response = answer_filter_question(question, df)

    print_filter_report(response)
