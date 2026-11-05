
from query_interpreter import interpret_question
from query_engine import execute_query
from result_explainer import explain_result


def answer_question(question, df):
    """
    Complete natural-language dataset querying workflow.

    1. Interpret the question using local Qwen.
    2. Execute the validated query using Python.
    3. Explain the calculated result using local Qwen.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Interpret the question.
    query_plan = interpret_question(question, df)

    # Step 2: Execute the query using Python.
    query_result = execute_query(df, query_plan)

    # Step 3: Explain the calculated result.
    explanation = explain_result(
        question,
        query_plan,
        query_result,
    )

    return {
        "question": question,
        "query_plan": query_plan,
        "query_result": query_result,
        "explanation": explanation,
    }


def print_answer_report(response):
    """
    Display the question, query plan,
    calculated result and AI explanation.
    """

    print("\n" + "=" * 50)
    print("DATASLEUTH INTELLIGENT QUERY")
    print("=" * 50)

    print("\nQuestion:")
    print(response["question"])

    print("\nQuery plan:")
    for key, value in response["query_plan"].items():
        print(f"{key}: {value}")

    print("\nCalculated result:")

    result = response["query_result"]["result"]

    if isinstance(result, list):
        for row in result:
            print(row)
    else:
        print(result)

    print("\nAI Explanation:")
    print(response["explanation"])

    print("\n" + "=" * 50)


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    question = "Which product generated the highest revenue?"

    response = answer_question(question, df)

    print_answer_report(response)
