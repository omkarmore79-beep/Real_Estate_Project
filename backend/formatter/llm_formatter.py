import os

from groq import Groq

from formatter.prompt_templates import REAL_ESTATE_PROMPT


_client = None


def _get_groq_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env and restart the server."
            )
        _client = Groq(api_key=api_key)
    return _client


def format_with_llm(text: str) -> str:
    """
    Use Groq LLM to extract structured real-estate metadata from raw PDF text.

    Returns a JSON string (may be wrapped in ``` fences — caller must strip those).
    Returns a minimal fallback JSON on any error so upload never crashes.
    """
    # Use the model configured in .env, defaulting to the verified Groq model
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    try:
        trimmed_text = text[:3000]
        prompt = REAL_ESTATE_PROMPT.format(input_text=trimmed_text)

        response = _get_groq_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        if answer:
            return answer

    except Exception as exc:
        print(f"[LLM FORMATTER] Error calling Groq ({model}): {exc}")

    # Fallback — returns valid JSON so json.loads() in the caller won't crash
    return '{"project_name": "Unknown", "details": "Could not extract metadata"}'
