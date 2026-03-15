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
from dotenv import load_dotenv

from utils.llm import summarize
from utils.formatters import format_summary
from utils.helpers import with_retry

# Configuration
OUTPUT_DIR           = Path("output")
PROMPT_FILE          = "prompt.txt"
LLM_MODEL_DEFAULT    = "gpt-4o-mini"
MAX_TRANSCRIPT_CHARS = 12_000
MAX_COMMENTS_CHARS   = 8_000

load_dotenv()

def _load_prompts():
    try:
        raw = Path(PROMPT_FILE).read_text(encoding="utf-8")
        parts = raw.split("## USER", 1)
        system = parts[0].replace("## SYSTEM", "").strip()
        user = parts[1].strip() if len(parts) > 1 else None
        return system, user
    except FileNotFoundError:
        return None, None

def strip_banner(text: str) -> str:
    """Strips the ASCII banner from the start of a transcript/summary if it exists."""
    if "======" in text:
        parts = text.split("========================================================================\n\n", 1)
        return parts[1] if len(parts) > 1 else text
    return text

def main():
    parser = argparse.ArgumentParser(description="Batch summarize extracted YouTube data.")
    parser.add_argument("--video", help="Specific Video ID to summarize (defaults to all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing summaries")
    parser.add_argument("--model", default=None, help=f"LLM model to use")
    parser.add_argument("--key-env", default="LLM_API_KEY", help="Env var name for the API key (default: LLM_API_KEY)")
    args = parser.parse_args()

    api_key = os.getenv(args.key_env)
    if not api_key:
        # Fallback to GOOGLE_API_KEY if LLM_API_KEY is missing
        if args.key_env == "LLM_API_KEY":
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                print(f"[INFO] LLM_API_KEY not found, using GOOGLE_API_KEY.")
    
    if not api_key:
        print(f"[ERR] API key not found in env var '{args.key_env}'. Summarization aborted.")
        print("      Please set it in .env or use --key-env <VAR_NAME>")
        sys.exit(1)

    # Determine model
    if args.model:
        model = args.model
    elif os.getenv("LLM_MODEL"):
        model = os.getenv("LLM_MODEL")
    else:
        # Smart default based on key type
        if "GOOGLE" in args.key_env or (args.key_env == "LLM_API_KEY" and os.getenv("GOOGLE_API_KEY")):
            model = "gemini-1.5-flash-latest"
        else:
            model = LLM_MODEL_DEFAULT

    system_p, user_p = _load_prompts()
    
    # Identify video directories
    if args.video:
        v_dirs = [OUTPUT_DIR / args.video]
    else:
        v_dirs = [d for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]

    if not v_dirs:
        print(f"[ERR] No extracted data found in {OUTPUT_DIR}/")
        return

    print(f"[SUMMARIZE] Using model: {model}")
    print(f"[SUMMARIZE] Processing {len(v_dirs)} video(s)...")

    results = {"ok": 0, "skip": 0, "fail": 0}

    import openai
    p_exc = (openai.AuthenticationError,)

    for v_dir in v_dirs:
        video_id = v_dir.name
        t_path = v_dir / "transcript.txt"
        c_path = v_dir / "comments.json"
        s_path = v_dir / "summary.txt"

        if not t_path.exists() or not c_path.exists():
            # Only skip silently if the folder is empty/invalid
            if any(v_dir.iterdir()):
                print(f"  [SKIP] {video_id} — missing data (transcript or comments)")
            results["skip"] += 1
            continue

        if s_path.exists() and not args.force:
            print(f"  [SKIP] {video_id} — summary already exists")
            results["skip"] += 1
            continue

        print(f"  [START] {video_id}")
        
        try:
            transcript = strip_banner(t_path.read_text(encoding="utf-8"))
            c_content = c_path.read_text(encoding="utf-8")
            if not c_content.strip():
                 print(f"  [FAIL] {video_id}: comments.json is empty")
                 results["fail"] += 1
                 continue
                 
            c_data = json.loads(c_content)
            comments = c_data.get("comments", [])
            url = c_data.get("meta", {}).get("url", f"https://youtube.com/watch?v={video_id}")

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
                print(f"  [FAIL] {video_id}: {err}")
                results["fail"] += 1
                continue

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

            print(f"  [OK] {video_id}")
            results["ok"] += 1

        except Exception as e:
            print(f"  [ERR] {video_id}: {e}")
            results["fail"] += 1

    print("-" * 30)
    print(f"[DONE] Summarized: {results['ok']} | Skipped: {results['skip']} | Failed: {results['fail']}")

if __name__ == "__main__":
    main()
