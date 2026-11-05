
import json
import re

from query_engine import prepare_dataset, validate_query_plan


MODEL = "qwen3:8b"


def build_prompt(question, df):
    """
    Build instructions for Qwen using the current dataset schema.
    """
    prepared_df = prepare_dataset(df)

    columns = {
        column: str(dtype)
        for column, dtype in prepared_df.dtypes.items()
    }

    return f"""
You are the query planner for DataSleuth, a local data analysis application.

Your job is to translate the user's question into a structured JSON query plan.

AVAILABLE DATASET COLUMNS:
{json.dumps(columns, indent=2)}

ALLOWED OPERATIONS:
- operation: "aggregate"

ALLOWED AGGREGATIONS:
- sum
- mean
- median
- min
- max
- count

QUERY PLAN FIELDS:
- operation: required
- metric: required dataset column
- aggregation: required allowed aggregation
- group_by: optional dataset column
- sort_by: optional metric or group_by column
- sort_order: optional "asc" or "desc"
- limit: optional positive integer

IMPORTANT RULES:
1. Return ONLY valid JSON.
2. Never generate Python code or SQL.
3. Never invent column names.
4. Never invent dataset values.
5. Use only the allowed operations and aggregations.
6. For revenue questions, use the "revenue" column.
7. For highest/top questions, use descending sorting.
8. For lowest/bottom questions, use ascending sorting.
9. If the question cannot be represented using the allowed operations,
   return an empty JSON object: {{}}.
10. Do not include explanations outside the JSON.

EXAMPLE:

User question:
Which product generated the highest revenue?

JSON response:
{{
    "operation": "aggregate",
    "metric": "revenue",
    "aggregation": "sum",
    "group_by": "product",
    "sort_by": "revenue",
    "sort_order": "desc",
    "limit": 1
}}

USER QUESTION:
{question}
"""


def extract_response_content(response):
    """
    Extract response text from Ollama's response object.
    Supports dictionary and object-style responses.
    """
    if isinstance(response, dict):
        message = response.get("message", {})
        content = message.get("content", "")
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen returned an empty response.")

    return content.strip()


def parse_query_plan(response_text):
    """
    Convert Qwen's JSON response into a Python dictionary.
    """
    if not isinstance(response_text, str):
        raise ValueError("AI response must be a string.")

    cleaned = response_text.strip()

    # Remove Markdown code fences if the model includes them.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen returned invalid JSON.") from exc

    if not isinstance(plan, dict):
        raise ValueError("Query plan must be a JSON object.")

    if not plan:
        raise ValueError("Qwen could not create a supported query plan.")

    return plan


def interpret_question(question, df, model=MODEL):
    """
    Ask local Qwen to interpret a natural-language question.

    The resulting plan is validated before it is returned.
    No generated code is executed.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    prepared_df = prepare_dataset(df)
    prompt = build_prompt(question, prepared_df)

    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON query planner. "
                        "Never generate executable code."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            format="json",
            options={
                "temperature": 0,
            },
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not communicate with local Ollama model: {exc}"
        ) from exc

    response_text = extract_response_content(response)
    plan = parse_query_plan(response_text)

    # Validate before the plan reaches the execution stage.
    validate_query_plan(prepared_df, plan)

    return plan


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    question = "Which product generated the highest revenue?"

    print("\nDATASLEUTH QUERY INTERPRETER")
    print("=" * 40)
    print("Question:", question)

    plan = interpret_question(question, df)

    print("\nGenerated query plan:")
    print(json.dumps(plan, indent=2))
