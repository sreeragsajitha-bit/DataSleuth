
import json
import re

import pandas as pd
import ollama

from query_engine import prepare_dataset
from filter_engine import validate_filters


MODEL = "qwen3:8b"

ALLOWED_OPERATORS = {
    "eq": "equal to",
    "ne": "not equal to",
    "gt": "greater than",
    "gte": "greater than or equal to",
    "lt": "less than",
    "lte": "less than or equal to",
    "contains": "text contains",
    "in": "value is in a list",
    "not_in": "value is not in a list",
    "is_null": "value is missing",
    "not_null": "value is not missing",
}


def build_filter_prompt(question, df):
    """Build a prompt containing the question and dataset information."""

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

    operators = "\n".join(
        f"{operator}: {description}"
        for operator, description in ALLOWED_OPERATORS.items()
    )

    return f"""
You are a database filter interpreter.

Convert the user's natural-language question into a JSON filter plan.

USER QUESTION:
{question}

DATASET COLUMNS AND OBSERVED VALUES:
{schema}

AVAILABLE OPERATORS:
{operators}

IMPORTANT RULES:

1. Select columns based on their observed values.
2. If the user mentions a category such as Accessories,
   select the column whose observed values contain Accessories.
3. Do not confuse product names with product categories.
4. For "accessories products", use the category column
   if its observed values include Accessories.
5. For a specific product such as Laptop, use the product column.
6. Preserve the intended filter values.
7. Use AND when all conditions must be satisfied.
8. Use OR only when the user explicitly requests alternatives.
9. Never generate Python code or SQL.
10. Never invent columns or operators.
11. Return only valid JSON.

Required JSON structure:

{{
    "filters": [
        {{
            "column": "category",
            "operator": "eq",
            "value": "Accessories"
        }},
        {{
            "column": "quantity",
            "operator": "gt",
            "value": 1
        }}
    ],
    "logical_operator": "and"
}}
"""


def extract_response_content(response):
    """Extract response text from Ollama dictionary or response objects."""

    if isinstance(response, dict):
        message = response.get("message")

        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = None

    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response.")

    return content.strip()


def parse_filter_plan(response):
    """Parse and validate the JSON structure returned by the LLM."""

    if not isinstance(response, str):
        raise ValueError("Response must be a string.")

    cleaned = response.strip()

    if not cleaned:
        raise ValueError("Ollama returned an empty response.")

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        plan = json.loads(cleaned)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON in filter plan: {error}"
        ) from error

    if not isinstance(plan, dict):
        raise ValueError("Filter plan must be a JSON object.")

    if not plan.get("filters"):
        raise ValueError(
            "could not create a filter plan with at least one condition."
        )

    return plan


def validate_filter_plan(df, plan):
    """Validate the filter plan without executing it."""

    if not isinstance(plan, dict):
        raise ValueError("Filter plan must be a JSON object.")

    allowed_keys = {"filters", "logical_operator"}

    unsupported_keys = set(plan.keys()) - allowed_keys

    if unsupported_keys:
        raise ValueError(
            f"Unsupported filter plan fields: {sorted(unsupported_keys)}"
        )

    filters = plan.get("filters")

    if not isinstance(filters, list) or not filters:
        raise ValueError(
            "Filter plan must contain at least one condition."
        )

    logical_operator = str(
        plan.get("logical_operator", "and")
    ).strip().lower()

    if logical_operator not in {"and", "or"}:
        raise ValueError(
            "Logical operator must be AND or OR."
        )

    for condition in filters:

        if not isinstance(condition, dict):
            raise ValueError(
                "Each filter condition must be an object."
            )

        column = condition.get("column")

        if column not in df.columns:
            raise ValueError(
                f"Unknown filter column: {column}"
            )

        operator = condition.get("operator")

        if operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Unsupported filter operator: {operator}"
            )

    validate_filters(df, filters, logical_operator)

    return True


def interpret_filter_question(question, df):
    """Convert a natural-language question into a validated filter plan."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    prepared_df = prepare_dataset(df)

    prompt = build_filter_prompt(question, prepared_df)

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert natural-language data questions into "
                    "accurate JSON filter plans. "
                    "Never generate Python or SQL code. "
                    "Return JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={"temperature": 0}
    )

    content = extract_response_content(response)

    plan = parse_filter_plan(content)

    validate_filter_plan(prepared_df, plan)

    plan["logical_operator"] = str(
        plan.get("logical_operator", "and")
    ).strip().lower()

    return plan


if __name__ == "__main__":
    from dataset_loader import load_dataset

    df = load_dataset("data/raw/sales.csv")

    question = (
        "Show accessories products where quantity is greater than 1."
    )

    result = interpret_filter_question(question, df)

    print("\nGenerated filter plan:")
    print(json.dumps(result, indent=4))
