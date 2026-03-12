"""
compare_models.py
-----------------
Runs multiple LLM models on existing transcript and comment data 
to compare their summarization performance side-by-side.
Reads model/API key mapping from models.txt.
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
OUTPUT_DIR  = Path("output")
PROMPT_FILE = "prompt.txt"
MODELS_FILE = "models.txt"

load_dotenv()

def load_prompts():
    try:
        raw = Path(PROMPT_FILE).read_text(encoding="utf-8")
        parts = raw.split("## USER", 1)
        system = parts[0].replace("## SYSTEM", "").strip()
        user = parts[1].strip() if len(parts) > 1 else None
        return system, user
    except FileNotFoundError:
        return None, None

def load_models_map():
    path = Path(MODELS_FILE)
    if not path.exists():
        print(f"[ERR] {MODELS_FILE} not found. Create it with: <model_name> <api_key_env_var>")
        sys.exit(1)
    
    models = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            # Model name is everything before the last word (API key env var)
            models.append((" ".join(parts[:-1]), parts[-1]))
    if not models:
        print(f"[ERR] No models found in {MODELS_FILE}")
        sys.exit(1)
    return models

def strip_banner(text: str) -> str:
    """Strips the ASCII banner from the start of a file if it exists."""
    if "======" in text:
        parts = text.split("========================================================================\n\n", 1)
        return parts[1] if len(parts) > 1 else text
    return text

def main():
    parser = argparse.ArgumentParser(description="Compare LLM models on YouTube data.")
    parser.add_argument("--video", help="Video ID to analyze (defaults to all)")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    args = parser.parse_args()

    models_map = load_models_map()
    system_p, user_p = load_prompts()
    
    v_dirs = [OUTPUT_DIR / args.video] if args.video else [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    if args.limit and not args.video:
        v_dirs = v_dirs[:args.limit]
        
    if not v_dirs:
        print("[ERR] No video data found in output/")
        return

    # Create comparisons directoy
    COMP_DIR = Path("comparisons")
    COMP_DIR.mkdir(exist_ok=True)

    for v_dir in v_dirs:
        v_id = v_dir.name
        t_path, c_path = v_dir / "transcript.txt", v_dir / "comments.json"
        
        if not t_path.exists() or not c_path.exists():
            continue

        print(f"\n[COMPARE] {v_id}")
        transcript = strip_banner(t_path.read_text(encoding="utf-8"))
        
        try:
            c_data = json.loads(c_path.read_text(encoding="utf-8"))
            comments = c_data.get("comments", [])
            url = c_data.get("meta", {}).get("url", f"https://youtube.com/watch?v={v_id}")
        except Exception:
            continue

        results = {}
        for model, key_name in models_map:
            api_key = os.getenv(key_name)
            if not api_key:
                print(f"  [SKIP] {model} ({key_name}): Key not set in .env")
                continue

            # Display name to distinguish same model with different keys
            display_name = f"{model} [{key_name}]"
            print(f"  -> {display_name}...", end="", flush=True)
            
            try:
                import openai
                p_exc = (openai.AuthenticationError,)
                # Call summarize with the actual model ID
                summary, err = with_retry(summarize, transcript, comments, api_key, model, system_p, user_p, label=display_name, permanent_exceptions=p_exc)
                if err:
                    print(f"\n  [ERR] {display_name}: {err}")
                    continue
                
                print(" OK")
                t_words = len(transcript.split())
                formatted = format_summary(summary, v_id, url, model, t_words, len(comments))
                
                # Sanitize filename
                safe_name = display_name.replace("/", "_").replace(":", "_").replace(" ", "_")
                (COMP_DIR / f"{v_id}_{safe_name}.txt").write_text(formatted, encoding="utf-8")
                results[display_name] = summary
            except Exception as e:
                print(f"  [ERR] {display_name}: {e}")

        if results:
            report_path = COMP_DIR / f"report_{v_id}.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Model Comparison: {v_id}\n\n**URL:** {url}\n\n")
                for m, res in results.items():
                    f.write(f"## {m}\n\n{res.strip()}\n\n---\n\n")
            print(f"  [OK] Centralized results in: {COMP_DIR}/{report_path.name}")

if __name__ == "__main__":
    main()
