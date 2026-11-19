from collections.abc import Iterator

from .config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_PROVIDER,
    MAX_HISTORY_MESSAGES,
    MAX_OUTPUT_TOKENS,
)

SYSTEM_PROMPT = """You are a friendly, conversational assistant that helps people understand their documents.

Answer in a warm, natural tone — like a knowledgeable friend explaining something, not a formal report.
- Write in flowing prose. No bullet points, no numbered lists, no markdown bold or asterisks.
- Never include citation markers like [1] or [2] in your answer.
- Keep answers concise and clear. Two to four sentences is usually enough.
- If the documents don't contain the answer, just say so simply and naturally.
- Base your answer only on the provided excerpts. This reduces the chance of incorrect information, but always verify important facts from the original source.
- Never make up information or use knowledge outside the provided excerpts."""


def _format_context(hits: list[dict]) -> str:
    if not hits:
        return "(no documents have been uploaded)"
    parts = []
    for i, hit in enumerate(hits, start=1):
        parts.append(
            f'<excerpt id="{i}" source="{hit["source"]}" page="{hit["page"]}">\n'
            f'{hit["text"]}\n</excerpt>'
        )
    return "\n\n".join(parts)


def _build_messages(question: str, hits: list[dict], history: list[dict]) -> list[dict]:
    """Recent history plus the new question with retrieved excerpts attached."""
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-MAX_HISTORY_MESSAGES:]
        if m["role"] in ("user", "assistant") and m["content"].strip()
    ]
    # Conversations must start with a user turn.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    messages.append(
        {
            "role": "user",
            "content": f"<excerpts>\n{_format_context(hits)}\n</excerpts>\n\n"
            f"Question: {question}",
        }
    )
    return messages


# ---------- Gemini ----------

_gemini_client = None


def _stream_gemini(messages: list[dict]) -> Iterator[str]:
    global _gemini_client
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(free key: https://aistudio.google.com/apikey)."
        )
    from google import genai
    from google.genai import types

    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    for chunk in _gemini_client.models.generate_content_stream(
        model=GEMINI_MODEL, contents=contents, config=config
    ):
        if chunk.text:
            yield chunk.text


# ---------- Claude (optional) ----------

_claude_client = None


def _stream_claude(messages: list[dict]) -> Iterator[str]:
    global _claude_client
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Run `pip install anthropic` to use LLM_PROVIDER=claude.")

    if _claude_client is None:
        _claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    with _claude_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_answer(
    question: str, hits: list[dict], history: list[dict]
) -> Iterator[str]:
    """Yield the answer from the configured LLM, piece by piece."""
    messages = _build_messages(question, hits, history)
    if LLM_PROVIDER == "claude":
        yield from _stream_claude(messages)
    elif LLM_PROVIDER == "gemini":
        yield from _stream_gemini(messages)
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'gemini' or 'claude'."
        )
