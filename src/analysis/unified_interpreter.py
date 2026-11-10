
import json
import re

import pandas as pd

from query_engine import prepare_dataset
from unified_analytics import validate_unified_plan


MODEL = "qwen3:8b"


def build_unified_prompt(question, df):
    """
    Build a prompt that teaches Qwen the dataset schema
    and the unified analytics plan format.
    """

    prepared_df = prepare_dataset(df)

    columns = {}

    for column in prepared_df.columns:
        series = prepared_df[column]

        column_info = {
            "dtype": str(series.dtype)
        }

        if (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            values = (
                series.dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            column_info["observed_values"] = values[:30]

        columns[column] = column_info

    schema = json.dumps(columns, indent=2, default=str)

    return f"""
You are the unified analytics planner for DataSleuth.

Convert the user's natural-language data question into
ONE structured JSON analytics plan.

DATASET COLUMNS AND OBSERVED VALUES:
{schema}

YOUR JOB:
Identify the filters and the aggregation needed to answer
the user's question.

FILTER PLAN FIELDS:

- filters: a list of filtering conditions.
- logical_operator: "and" or "or".

Each filter condition must contain:
- column: an existing dataset column.
- operator: an allowed filter operator.
- value: the comparison value, when required.

ALLOWED FILTER OPERATORS:
- eq
- ne
- gt
- gte
- lt
- lte
- contains
- in
- not_in
- is_null
- not_null

QUERY PLAN FIELDS:

- operation: must be "aggregate".
- metric: an existing dataset column.
- aggregation: an allowed aggregation.
- group_by: optional existing dataset column.
- sort_by: optional metric or group_by column.
- sort_order: optional "asc" or "desc".
- limit: optional positive integer.

ALLOWED AGGREGATIONS:
- sum
- mean
- median
- min
- max
- count

IMPORTANT RULES:

1. Return ONLY valid JSON.
2. Never generate Python code or SQL.
3. Never invent dataset columns or values.
4. Use observed values to identify the correct filter column.
5. Do not confuse product names with product categories.
6. Use AND when all filter conditions must be satisfied.
7. Use OR only when the user explicitly requests alternatives.
8. Use an empty filters list when no filtering is needed.
9. Use "revenue" for revenue questions when available.
10. Use descending sorting for highest or top questions.
11. Use ascending sorting for lowest or bottom questions.
12. Use group_by when the question compares categories or groups.
13. Use limit for requests such as top 5 or bottom 3.
14. Do not add fields outside the required JSON structure.
15. If the question cannot be represented using these operations,
    return an empty JSON object: {{}}.

REQUIRED JSON STRUCTURE:

{{
    "filters": [],
    "logical_operator": "and",
    "operation": "aggregate",
    "metric": "revenue",
    "aggregation": "sum",
    "group_by": "product",
    "sort_by": "revenue",
    "sort_order": "desc",
    "limit": 2
}}

The group_by, sort_by, sort_order and limit fields are optional.
Use only the fields needed to answer the question.

USER QUESTION:
{question}
"""


def extract_response_content(response):
    """
    Extract response text from an Ollama dictionary
    or object-style response.
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


def parse_unified_plan(response_text):
    """
    Convert Qwen's JSON response into a Python dictionary.
    """

    if not isinstance(response_text, str):
        raise ValueError("AI response must be a string.")

    cleaned = response_text.strip()

    if not cleaned:
        raise ValueError("Qwen returned an empty response.")

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen returned invalid JSON.") from exc

    if not isinstance(plan, dict):
        raise ValueError("Unified plan must be a JSON object.")

    if not plan:
        raise ValueError(
            "Qwen could not create a supported unified plan."
        )

    return plan


def interpret_unified_question(question, df, model=MODEL):
    """
    Convert a natural-language question into a validated
    unified analytics plan using local Qwen.

    Qwen proposes the plan.
    Python validates the plan.
    No generated code is executed.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    prepared_df = prepare_dataset(df)

    prompt = build_unified_prompt(
        question,
        prepared_df,
    )

    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON analytics planner. "
                        "Return only a valid unified analytics plan. "
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

    plan = parse_unified_plan(response_text)

    # Validate filters and aggregation before returning the plan.
    validate_unified_plan(
        prepared_df,
        plan,
    )

    return plan
