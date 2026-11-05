
import json

MODEL = "qwen3:8b"


def build_explanation_prompt(question, query_plan, query_result):
    """
    Build a prompt using the original question,
    validated query plan and Python-calculated result.
    """

    evidence = {
        "question": question,
        "query_plan": query_plan,
        "calculated_result": query_result,
    }

    return f"""
You are DataSleuth, an AI data analysis assistant.

Explain the result of a dataset query in clear, natural language.

The following information comes from the trusted Python analysis engine:

{json.dumps(evidence, indent=2, default=str)}

STRICT RULES:

1. Explain only what the calculated result demonstrates.
2. Never invent statistics, numbers, trends or comparisons.
3. Never modify or recalculate the supplied result.
4. Do not claim causation or relationships that were not tested.
5. Do not introduce external information.
6. If the result is empty, clearly say that no matching results were found.
7. Keep the explanation concise and understandable.
8. Do not mention internal implementation details unless asked.
9. Do not assume a currency symbol or unit unless the dataset provides it.

USER QUESTION:
{question}

Write a concise answer based strictly on the supplied evidence.
"""


def extract_explanation(response):
    """
    Extract explanation text from an Ollama response.
    Supports dictionary and object-style responses.
    """

    if isinstance(response, dict):
        message = response.get("message", {})
        content = message.get("content", "")
    else:
        message = getattr(response, "message", None)
        content = getattr(message, "content", "")

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen returned an empty explanation.")

    return content.strip()


def explain_result(question, query_plan, query_result, model=MODEL):
    """
    Generate a natural-language explanation using local Qwen.

    Qwen receives the result from Python.
    It does not execute queries or modify the calculations.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(query_plan, dict):
        raise ValueError("Query plan must be a dictionary.")

    if not isinstance(query_result, dict):
        raise ValueError("Query result must be a dictionary.")

    if query_result.get("success") is not True:
        raise ValueError("Cannot explain an unsuccessful query.")

    prompt = build_explanation_prompt(
        question,
        query_plan,
        query_result,
    )

    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful data analyst. "
                        "Explain only the supplied calculated evidence. "
                        "Never invent or alter numerical results."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not communicate with local Ollama model: {exc}"
        ) from exc

    return extract_explanation(response)
