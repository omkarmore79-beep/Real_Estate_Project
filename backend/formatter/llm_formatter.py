import os

from groq import Groq

from formatter.prompt_templates import REAL_ESTATE_PROMPT


client = None


def _get_groq_client():
    global client
    if client is None:
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
    return client


def format_with_llm(text):
    try:
        trimmed_text = text[:3000]
        prompt = REAL_ESTATE_PROMPT.format(
            input_text=trimmed_text
        )

        response = _get_groq_client().chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        answer = response.choices[0].message.content
        if answer:
            return answer

        return """
        {
            "project_name": "Unknown",
            "details": "No response from Groq"
        }
        """

    except Exception as e:
        print("LLM FORMATTER ERROR:", e)

        return """
        {
            "project_name": "Unknown",
            "details": "Groq API error"
        }
        """
