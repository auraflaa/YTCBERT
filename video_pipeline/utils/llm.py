import openai
from google import genai
from utils.stats import comment_texts

DEFAULT_SYSTEM_PROMPT = "You analyze YouTube video discussions. Be concise and structured."
DEFAULT_USER_PROMPT = (
    "TRANSCRIPT:\n{transcript}\n\n"
    "TOP COMMENTS:\n{comments}\n\n"
    "Summarize viewer sentiment and key themes, referencing the transcript where relevant."
)

TRANSCRIPT_SUMMARY_PROMPT = (
    "Summarize the following transcript segment concisely, focusing on the main topics and key points discussed. "
    "Keep it under 300 words.\n\n"
    "TRANSCRIPT SEGMENT:\n{transcript}"
)

COMMENT_SYNTHESIS_PROMPT = (
    "VIDEO CONTEXT (SUMMARIZED TRANSCRIPT):\n{context}\n\n"
    "COMMENTS BATCH:\n{comments}\n\n"
    "Based on the video context provided, summarize the core sentiment, unique perspectives, and common questions in this batch of comments. "
    "Be concise."
)

FINAL_AGGREGATION_PROMPT = (
    "VIDEO CONTEXT (SUMMARIZED TRANSCRIPT):\n{context}\n\n"
    "INTERMEDIATE COMMENT SUMMARIES:\n{summaries}\n\n"
    "Create a comprehensive final summary of the entire video and its community discussion. "
    "Identify overarching themes and the ultimate consensus from the viewers."
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
        target_model = model if model.startswith("models/") else f"models/{model.lower().replace(' ', '-')}"
        
        # Check if the model supports system_instruction (standard Gemini does, Gemma often doesn't)
        is_gemma = "gemma" in target_model.lower()
        
        if is_gemma:
            # For Gemma, prepend system prompt to user message instead of using system_instruction
            full_msg = f"{sys_p}\n\n{user_msg}"
            resp = client.models.generate_content(
                model=target_model,
                contents=full_msg
            )
        else:
            resp = client.models.generate_content(
                model=target_model,
                config={'system_instruction': sys_p},
                contents=user_msg
            )
        return resp.text
    else:
        raise ValueError(f"Unsupported model: {model}. Must start with 'gpt-', 'gemini-', or contain 'gemma'.")


def summarize_transcript_chunk(transcript: str, api_key: str, model: str) -> str:
    """Helper for Stage 1: Condensing transcript chunks."""
    user_p = TRANSCRIPT_SUMMARY_PROMPT.format(transcript=transcript)
    return call_llm(DEFAULT_SYSTEM_PROMPT, user_p, api_key, model)


def summarize_comment_batch(context: str, comments_text: str, api_key: str, model: str) -> str:
    """Helper for Stage 2: Synthesizing comment batches with transcript context."""
    user_p = COMMENT_SYNTHESIS_PROMPT.format(context=context, comments=comments_text)
    return call_llm(DEFAULT_SYSTEM_PROMPT, user_p, api_key, model)


def summarize_final_aggregation(context: str, summaries_text: str, api_key: str, model: str) -> str:
    """Helper for Stage 3: Aggregating all intermediate summaries into a master summary."""
    user_p = FINAL_AGGREGATION_PROMPT.format(context=context, summaries=summaries_text)
    return call_llm(DEFAULT_SYSTEM_PROMPT, user_p, api_key, model)


def call_llm(system_p: str, user_p: str, api_key: str, model: str) -> str:
    """Generic wrapper for direct LLM calls used by hierarchical stages."""
    if model.startswith("gpt-") or model.startswith("openrouter/"):
        base_url = "https://openrouter.ai/api/v1" if model.startswith("openrouter/") else None
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        target_model = model.replace("openrouter/", "") if model.startswith("openrouter/") else model
        resp = client.chat.completions.create(
            model=target_model,
            messages=[
                {"role": "system", "content": system_p},
                {"role": "user",   "content": user_p},
            ],
        )
        return resp.choices[0].message.content
    elif any(x in model.lower() for x in ["gemini", "gemma"]) or model.startswith("models/"):
        client = genai.Client(api_key=api_key)
        target_model = model if model.startswith("models/") else f"models/{model.lower().replace(' ', '-')}"
        is_gemma = "gemma" in target_model.lower()
        if is_gemma:
            full_msg = f"{system_p}\n\n{user_p}"
            resp = client.models.generate_content(model=target_model, contents=full_msg)
        else:
            resp = client.models.generate_content(
                model=target_model,
                config={'system_instruction': system_p},
                contents=user_p
            )
        return resp.text
    return f"Error: Unsupported model {model}"
