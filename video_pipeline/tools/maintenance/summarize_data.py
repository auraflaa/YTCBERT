"""
tools/maintenance/summarize_data.py
----------------------------------
Hierarchical AI Summarization Core. Processes dataset using Map-Reduce.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from dotenv import load_dotenv

# Local Imports
try:
    from utils.llm import summarize_transcript_chunk, summarize_comment_batch, summarize_final_aggregation
    from utils.formatters import format_summary
    from utils.helpers import with_retry, strip_banner, chunk_text, chunk_list
except ImportError:
    print(f"Error: Could not find 'utils' module. ROOT_DIR: {ROOT_DIR}")
    sys.exit(1)

# Configuration
OUTPUT_DIR              = ROOT_DIR / "output"
LLM_MODEL_DEFAULT       = "gpt-4o-mini"
MAX_TRANSCRIPT_CHUNK    = 10_000  
MAX_COMMENT_BATCH       = 500     
DEFAULT_WORKERS_OUTER   = 5
DEFAULT_WORKERS_INNER   = 3       

load_dotenv(ROOT_DIR / ".env")
console = Console()

def process_single_video(v_dir, api_key, model, force, workers_inner):
    """Orchestrates hierarchical summarization for a single video."""
    video_id = v_dir.name
    t_path   = v_dir / "transcript.txt"
    c_path   = v_dir / "comments.json"
    s_path   = v_dir / "summary.txt"

    if not t_path.exists() or not c_path.exists():
        return "skip", None
    if s_path.exists() and not force:
        return "skip", f"{video_id} — already summarized"

    try:
        # --- STAGE 1: Transcript Condensation ---
        raw_t = strip_banner(t_path.read_text(encoding="utf-8"))
        t_chunks = chunk_text(raw_t, MAX_TRANSCRIPT_CHUNK)
        chunk_results = []
        with ThreadPoolExecutor(max_workers=workers_inner) as exe:
            futures = [exe.submit(with_retry, summarize_transcript_chunk, ch, api_key, model, label=f"T-Chunk({video_id})") for ch in t_chunks]
            for f in as_completed(futures):
                res, err = f.result()
                if err: return "fail", f"Stage 1 failed: {err}"
                chunk_results.append(res)
        transcript_summary = "\n\n".join(chunk_results)

        # --- STAGE 2: Comment Synthesis ---
        c_data = json.loads(c_path.read_text(encoding="utf-8"))
        all_comments = c_data.get("comments", [])
        comment_batches = chunk_list(all_comments, MAX_COMMENT_BATCH)
        url = c_data.get("meta", {}).get("url", f"https://youtube.com/watch?v={video_id}")
        intermediate_summaries = []
        with ThreadPoolExecutor(max_workers=workers_inner) as exe:
            futures = []
            for batch in comment_batches:
                batch_text = "\n".join(f"- {c.get('text', '')}" for c in batch)
                futures.append(exe.submit(with_retry, summarize_comment_batch, transcript_summary, batch_text, api_key, model, label=f"C-Batch({video_id})"))
            for f in as_completed(futures):
                res, err = f.result()
                if err: return "fail", f"Stage 2 failed: {err}"
                intermediate_summaries.append(res)

        # --- STAGE 3: Final Aggregation ---
        summaries_block = "\n\n".join(intermediate_summaries)
        final_summary, err = with_retry(summarize_final_aggregation, transcript_summary, summaries_block, api_key, model, label=f"Final({video_id})")
        if err: return "fail", f"Stage 3 failed: {err}"

        raw_t = strip_banner(t_path.read_text(encoding="utf-8"))
        formatted = format_summary(final_summary, video_id, url, model, len(raw_t.split()), len(all_comments))
        s_path.write_text(formatted, encoding="utf-8")
        
        # Update meta status
        m_path = v_dir / "meta.json"
        if m_path.exists():
            try:
                meta = json.loads(m_path.read_text(encoding="utf-8"))
                if "status" not in meta: meta["status"] = {}
                meta["status"]["summary_hierarchical"] = "ok"
                m_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception: pass
        return "ok", video_id
    except Exception as e:
        return "fail", f"{video_id}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Summary Generation.")
    parser.add_argument("--video")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS_OUTER)
    parser.add_argument("--workers-inner", type=int, default=DEFAULT_WORKERS_INNER)
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print("[ERR] No API key found.")
        sys.exit(1)
    
    model = args.model if args.model else ("gemini-1.5-flash" if os.getenv("GOOGLE_API_KEY") else LLM_MODEL_DEFAULT)
    v_dirs = [OUTPUT_DIR / args.video] if args.video else [d for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]
    if not v_dirs:
        print("[ERR] No data found.")
        return

    console.print(f"[bold blue][HIERARCHICAL] Model: {model}[/bold blue]\n")
    results = {"ok": 0, "skip": 0, "fail": 0}
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), MofNCompleteColumn(), console=console, expand=True) as progress:
        task_id = progress.add_task("[cyan]Summarizing...", total=len(v_dirs))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, v_dir, api_key, model, args.force, args.workers_inner): v_dir for v_dir in v_dirs}
            try:
                for future in as_completed(futures):
                    outcome, msg = future.result()
                    results[outcome] += 1
                    if outcome == "fail": progress.console.print(f"  [red][FAIL] {msg}[/red]")
                    progress.advance(task_id)
            except KeyboardInterrupt:
                progress.console.print("\n[bold red][HALT] Interrupted.[/bold red]")
                for f in futures: f.cancel()
    console.print(f"[bold green]Finished![/bold green] OK: {results['ok']} | Fail: {results['fail']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[bold red][ERR] {e}[/bold red]")
        sys.exit(1)
