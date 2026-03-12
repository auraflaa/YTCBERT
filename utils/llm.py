import openai
from google import genai
from utils.stats import comment_texts

DEFAULT_SYSTEM_PROMPT = "You analyze YouTube video discussions. Be concise and structured."
DEFAULT_USER_PROMPT = (
    "TRANSCRIPT:\n{transcript}\n\n"
    "TOP COMMENTS ({n_comments}):\n{comments}\n\n"
    "Summarize viewer sentiment and key themes, referencing the transcript where relevant."
)

def summarize(
    transcript: str | None, 
    comments: list[dict], 
    api_key: str,
    model: str,
    system_prompt: str = None,
    user_prompt_template: str = None,
    max_transcript_chars: int = 12000,
    max_comments_chars: int = 8000
) -> str:
    """Formats the prompt and calls the LLM for summarization."""
    if not api_key:
        raise ValueError("API_KEY is not set.")

    # Fallback to defaults if prompts not provided
    sys_p = system_prompt or DEFAULT_SYSTEM_PROMPT
    usr_p = user_prompt_template or DEFAULT_USER_PROMPT

    # Process transcript
    t = (transcript or "(No transcript)")[:max_transcript_chars]
    if len(transcript or "") > max_transcript_chars:
        t += "\n...[transcript truncated]"

    # Process comments
    texts = comment_texts(comments)
    c_block = "\n".join(f"- {c}" for c in texts)[:max_comments_chars]
    if sum(len(c) + 2 for c in texts) > max_comments_chars:
        c_block += "\n...[comments truncated]"

    # Build final user message
    user_msg = usr_p.format(
        transcript=t, 
        n_comments=len(texts), 
        comments=c_block
    )

    if model.startswith("gpt-") or model.startswith("openrouter/"):
        # OpenAI or OpenRouter (OpenRouter uses OpenAI-compatible API)
        base_url = "https://openrouter.ai/api/v1" if model.startswith("openrouter/") else None
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        
        # OpenRouter wants the full model path if provided, e.g. "google/gemma-3-27b-it"
        target_model = model.replace("openrouter/", "") if model.startswith("openrouter/") else model
        
        resp = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user",   "content": user_msg},
            ],
        )
        return resp.choices[0].message.content
    elif any(x in model.lower() for x in ["gemini", "gemma"]) or model.startswith("models/"):
        # Google Gemini / Gemma
        client = genai.Client(api_key=api_key)
        # Handle "Gemma 3 27B" style names by making them standard slugs if needed
        # but modern SDK usually likes the string as is or with models/ prefix
        target_model = model if model.startswith("models/") else f"models/{model.lower().replace(' ', '-')}"
        
        resp = client.models.generate_content(
            model=target_model,
            config={'system_instruction': sys_p},
            contents=user_msg
        )
        return resp.text
    else:
        raise ValueError(f"Unsupported model: {model}. Must start with 'gpt-', 'gemini-', or contain 'gemma'.")
