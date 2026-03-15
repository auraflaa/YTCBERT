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
OUTPUT_DIR  = Path("output")
MODELS_FILE = "models.txt"
COMP_DIR    = Path("comparisons")
MAX_TRANSCRIPT_CHARS = 12_000
MAX_COMMENTS_CHARS   = 8_000

load_dotenv()

def load_models_map():
    path = Path(MODELS_FILE)
    if not path.exists():
        print(f"[ERR] {MODELS_FILE} not found. Create it with: <model_name> <api_key_env_var>")
        sys.exit(1)
    
    models = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            models.append((" ".join(parts[:-1]), parts[-1]))
    if not models:
        print(f"[ERR] No models found in {MODELS_FILE}")
        sys.exit(1)
    return models

def run_model_comparison(video_id, url, transcript, comments, model, key_name, system_p, user_p):
    """Runs a single model comparison for a video."""
    api_key = os.getenv(key_name)
    if not api_key:
        return None, f"{model} ({key_name}): Key not set in .env"

    display_name = f"{model} [{key_name}]"
    try:
        import openai
        p_exc = (openai.AuthenticationError,)
        
        summary, err = with_retry(
            summarize, 
            transcript, comments, api_key, model, 
            system_p, user_p, 
            MAX_TRANSCRIPT_CHARS,
            MAX_COMMENTS_CHARS,
            label=display_name, 
            permanent_exceptions=p_exc
        )
        
        if err:
            return None, f"{display_name}: {err}"
        
        t_words = len(transcript.split())
        formatted = format_summary(summary, video_id, url, model, t_words, len(comments))
        
        safe_name = display_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        (COMP_DIR / f"{video_id}_{safe_name}.txt").write_text(formatted, encoding="utf-8")
        
        return display_name, summary
    except Exception as e:
        return None, f"{display_name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Compare LLM models on YouTube data.")
    parser.add_argument("--video", help="Video ID to analyze (defaults to all)")
    parser.add_argument("--limit", type=int, help="Limit number of videos to process")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (default: 5)")
    args = parser.parse_args()

    models_map = load_models_map()
    system_p, user_p = load_prompts()
    
    v_dirs = [OUTPUT_DIR / args.video] if args.video else [d for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]
    if args.limit and not args.video:
        v_dirs = v_dirs[:args.limit]
        
    if not v_dirs:
        print("[ERR] No video data found in output/")
        return

    COMP_DIR.mkdir(exist_ok=True)

    for v_dir in v_dirs:
        video_id = v_dir.name
        t_path, c_path = v_dir / "transcript.txt", v_dir / "comments.json"
        
        if not t_path.exists() or not c_path.exists():
            continue

        print(f"\n[COMPARE] {video_id}")
        transcript = strip_banner(t_path.read_text(encoding="utf-8"))
        
        try:
            c_data = json.loads(c_path.read_text(encoding="utf-8"))
            comments = c_data.get("comments", [])
            url = c_data.get("meta", {}).get("url", f"https://youtube.com/watch?v={video_id}")
        except Exception:
            continue

        comparison_results = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_model_comparison, video_id, url, transcript, comments, m, k, system_p, user_p) for m, k in models_map]
            for future in as_completed(futures):
                name, result = future.result()
                if name:
                    print(f"  -> {name}: OK")
                    comparison_results[name] = result
                else:
                    print(f"  -> [ERR] {result}")

        if comparison_results:
            report_path = COMP_DIR / f"report_{video_id}.md"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# Model Comparison: {video_id}\n\n**URL:** {url}\n\n")
                for m, res in sorted(comparison_results.items()):
                    f.write(f"## {m}\n\n{res.strip()}\n\n---\n\n")
            print(f"  [OK] Centralized report: {COMP_DIR}/{report_path.name}")

if __name__ == "__main__":
    main()
