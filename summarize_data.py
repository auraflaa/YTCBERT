"""
summarize_data.py
-----------------
Batch processes extracted YouTube data (transcripts and comments) in the output/ 
directory and generates LLM-based summaries.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from utils.llm import summarize
from utils.formatters import format_summary
from utils.helpers import with_retry, strip_banner, load_prompts

# Configuration
OUTPUT_DIR           = Path("output")
PROMPT_FILE          = "prompt.txt"
LLM_MODEL_DEFAULT    = "gpt-4o-mini"
MAX_TRANSCRIPT_CHARS = 12_000
MAX_COMMENTS_CHARS   = 8_000
DEFAULT_WORKERS      = 5

load_dotenv()

def process_single_video(v_dir, api_key, model, system_p, user_p, force):
    """Processes a single video directory for summarization."""
    video_id = v_dir.name
    t_path = v_dir / "transcript.txt"
    c_path = v_dir / "comments.json"
    s_path = v_dir / "summary.txt"

    if not t_path.exists() or not c_path.exists():
        if any(v_dir.iterdir()):
             return "skip", f"{video_id} — missing data (transcript or comments)"
        return "skip", None

    if s_path.exists() and not force:
        return "skip", f"{video_id} — summary already exists"

    try:
        transcript = strip_banner(t_path.read_text(encoding="utf-8"))
        c_content = c_path.read_text(encoding="utf-8")
        if not c_content.strip():
             return "fail", f"{video_id}: comments.json is empty"
             
        c_data = json.loads(c_content)
        comments = c_data.get("comments", [])
        url = c_data.get("meta", {}).get("url", f"https://youtube.com/watch?v={video_id}")

        import openai
        p_exc = (openai.AuthenticationError,)

        summary, err = with_retry(
            summarize, 
            transcript, comments, api_key, model, 
            system_p, user_p, 
            MAX_TRANSCRIPT_CHARS,
            MAX_COMMENTS_CHARS,
            label=f"Summary({video_id})",
            permanent_exceptions=p_exc
        )

        if err:
            return "fail", f"{video_id}: {err}"

        t_words = len(transcript.split())
        formatted = format_summary(summary, video_id, url, model, t_words, len(comments))
        s_path.write_text(formatted, encoding="utf-8")
        
        # Update meta.json status
        m_path = v_dir / "meta.json"
        if m_path.exists():
            try:
                meta = json.loads(m_path.read_text(encoding="utf-8"))
                if "status" not in meta: meta["status"] = {}
                meta["status"]["summary"] = "ok"
                m_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        return "ok", video_id

    except Exception as e:
        return "fail", f"{video_id}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Batch summarize extracted YouTube data.")
    parser.add_argument("--video", help="Specific Video ID to summarize (defaults to all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing summaries")
    parser.add_argument("--model", default=None, help=f"LLM model to use")
    parser.add_argument("--key-env", default="LLM_API_KEY", help="Env var name for the API key (default: LLM_API_KEY)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of parallel workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    api_key = os.getenv(args.key_env)
    if not api_key:
        if args.key_env == "LLM_API_KEY":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                print(f"[INFO] LLM_API_KEY not found, using GOOGLE_API_KEY.")
    
    if not api_key:
        print(f"[ERR] API key not found in env var '{args.key_env}'. Summarization aborted.")
        sys.exit(1)

    # Determine model
    if args.model:
        model = args.model
    elif os.getenv("LLM_MODEL"):
        model = os.getenv("LLM_MODEL")
    else:
        if "GOOGLE" in args.key_env or (args.key_env == "LLM_API_KEY" and os.getenv("GOOGLE_API_KEY")):
            model = "gemini-1.5-flash-latest"
        else:
            model = LLM_MODEL_DEFAULT

    system_p, user_p = load_prompts(PROMPT_FILE)
    
    if args.video:
        v_dirs = [OUTPUT_DIR / args.video]
    else:
        v_dirs = [d for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]

    if not v_dirs:
        print(f"[ERR] No extracted data found in {OUTPUT_DIR}/")
        return

    print(f"[SUMMARIZE] Using model: {model} | Workers: {args.workers}")
    print(f"[SUMMARIZE] Processing {len(v_dirs)} video(s)...")

    results = {"ok": 0, "skip": 0, "fail": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_vdir = {executor.submit(process_single_video, v_dir, api_key, model, system_p, user_p, args.force): v_dir for v_dir in v_dirs}
        for future in as_completed(future_to_vdir):
            outcome, msg = future.result()
            results[outcome] += 1
            if outcome == "ok":
                print(f"  [OK] {msg}")
            elif msg:
                print(f"  [{outcome.upper()}] {msg}")

    print("-" * 30)
    print(f"[DONE] Summarized: {results['ok']} | Skipped: {results['skip']} | Failed: {results['fail']}")

if __name__ == "__main__":
    main()
