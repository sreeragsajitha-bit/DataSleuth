
import json

from ollama import chat

try:
    from src.analysis.claim_verifier import (
        verify_claims,
        create_verification_summary
    )
except ImportError:
    from analysis.claim_verifier import (
        verify_claims,
        create_verification_summary
    )


MODEL = "qwen3:8b"
MAX_REGENERATION_ATTEMPTS = 1


GENERAL_SYSTEM_PROMPT = """
You are DataSleuth, a local AI data science assistant.

Explain data science concepts clearly and accurately.
Use practical examples.
Distinguish correlation from causation.
Never pretend to have analysed a dataset unless analysis evidence
has actually been supplied.
Be beginner-friendly and technically accurate.
"""


STRUCTURED_SYSTEM_PROMPT = """
You are DataSleuth, an AI data analysis assistant.

Python is the source of truth for dataset-specific facts.

STRICT RULES:

1. Never invent numerical values.
2. Every dataset-specific factual claim must have supporting evidence.
3. Every claim must include its exact evidence_path.
4. Every claim must include the value found at that path.
5. Never confuse columns or statistical measurements.
6. Correlation does not establish causation.
7. Distinguish observations from hypotheses.
8. If evidence is unavailable, say so.
9. Include all dataset-specific factual claims in the claims array.
10. Do not omit claims to avoid verification.

OUTPUT FORMAT:

Return a valid JSON object:

{
    "answer": "Your explanation here.",
    "claims": [
        {
            "claim": "The age column has a mean of 24.375.",
            "evidence_path": "profile.column_profiles.age.statistics.mean",
            "value": 24.375
        }
    ]
}

Do not surround the JSON with Markdown code fences.
"""


def ask_ai(question):
    """Ask Qwen a general data science question."""

    if not isinstance(question, str) or not question.strip():
        return "Please enter a valid question."

    try:
        response = chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": GENERAL_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            options={"temperature": 0.2}
        )

        return response.message.content

    except Exception as error:
        return (
            "Unable to communicate with the local AI model.\n"
            f"Details: {error}"
        )


def prepare_analysis_context(results):
    """Convert Python analysis results into JSON context."""

    context = {
        "dataset_name": results.get("dataset_name"),
        "profile": results.get("profile", {}),
        "quality_issues": results.get("quality_issues", []),
        "insights": results.get("insights", [])
    }

    return json.dumps(context, indent=2, default=str)


def parse_ai_response(response_text):
    """Parse and validate Qwen's structured JSON response."""

    if not isinstance(response_text, str):
        raise ValueError("AI response must be text.")

    cleaned = response_text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)

    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError("AI did not return valid JSON.")

        result = json.loads(cleaned[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("AI response must be a JSON object.")

    if not isinstance(result.get("answer"), str):
        raise ValueError("AI response is missing a valid answer.")

    if not isinstance(result.get("claims"), list):
        raise ValueError("AI response is missing a valid claims list.")

    return result


def request_structured_analysis(context, question, previous_errors=None):
    """Ask Qwen to generate an evidence-backed explanation."""

    retry_instructions = ""

    if previous_errors:
        retry_instructions = f"""
YOUR PREVIOUS RESPONSE FAILED EVIDENCE VERIFICATION.

These claims were rejected:

{json.dumps(previous_errors, indent=2, default=str)}

Regenerate your answer.
Correct the evidence paths and values, or remove unsupported claims.
Do not repeat the verification errors.
"""

    user_prompt = f"""
DATASET ANALYSIS REPORT:

{context}

USER QUESTION:

{question}

For every dataset-specific factual claim:
- Include the claim in the claims array.
- Give its exact evidence_path.
- Give the value found at that path.

Evidence paths must refer to the supplied report.

{retry_instructions}

Return only a valid JSON object.
"""

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": STRUCTURED_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        format="json",
        options={"temperature": 0.1}
    )

    return parse_ai_response(response.message.content)


def analyze_with_ai(results, question=None):
    """
    Explain a Python-generated report using Qwen.

    Failed verification triggers one regeneration attempt.
    Remaining unsupported claims are explicitly flagged.
    """

    if not isinstance(results, dict):
        return "Invalid analysis report supplied."

    if "profile" not in results:
        return "Analysis report does not contain a dataset profile."

    if question is None:
        question = (
            "Analyse this dataset. Explain its important findings, "
            "potential problems and useful next steps."
        )

    if not isinstance(question, str) or not question.strip():
        return "Please enter a valid analysis question."

    context = prepare_analysis_context(results)

    previous_errors = None
    final_response = None
    final_verification = None

    for attempt in range(MAX_REGENERATION_ATTEMPTS + 1):

        try:
            final_response = request_structured_analysis(
                context=context,
                question=question,
                previous_errors=previous_errors
            )

        except Exception as error:
            if final_response is None:
                return (
                    "DataSleuth could not generate an AI explanation.\n"
                    f"Details: {error}"
                )

            break

        final_verification = verify_claims(
            report=results,
            claims=final_response["claims"]
        )

        if final_verification["all_verified"]:
            break

        previous_errors = final_verification["rejected"]

    if final_response is None:
        return "DataSleuth could not generate an AI explanation."

    answer = final_response["answer"]

    if final_verification is None:
        return (
            answer
            + "\n\nVERIFICATION STATUS: Unable to verify AI claims."
        )

    summary = create_verification_summary(final_verification)

    return (
        answer
        + "\n\n"
        + "=" * 45
        + "\n"
        + summary
    )


if __name__ == "__main__":
    print("DATASLEUTH LOCAL AI")
    print("=" * 40)

    question = input("\nAsk DataSleuth a question: ")

    print("\nDATASLEUTH RESPONSE")
    print("-" * 40)
    print(ask_ai(question))
