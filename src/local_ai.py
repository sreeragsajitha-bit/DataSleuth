
from ollama import chat

MODEL = "qwen3:8b"

SYSTEM_PROMPT = """
You are DataSleuth, a local AI data science assistant.

Your responsibilities:
1. Explain data science concepts clearly.
2. Help users understand datasets and statistics.
3. Assist with Python and machine learning.
4. Never invent numerical results.
5. Explain technical concepts with practical examples.

Keep responses structured and beginner-friendly.
"""


def ask_ai(question):
    if not question.strip():
        return "Please enter a valid question."

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.message.content


if __name__ == "__main__":
    print("===== DATASLEUTH AI =====")

    question = input("\nAsk DataSleuth: ")

    answer = ask_ai(question)

    print("\nDataSleuth says:\n")
    print(answer)
