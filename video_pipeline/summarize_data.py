"""
Hierarchical AI Summarization Core (Step 2).

Processes raw data in the output/ directory using a 3-stage Map-Reduce approach:
Stage 1 (Condensation): Parallel transcript chunk summarization.
Stage 2 (Synthesis): Comment batch summarization with transcript context.
Stage 3 (Aggregation): Final master summary generation.

Ensures long videos and thousands of comments fit within the LLM's context window.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn
from utils.llm import (
    summarize, summarize_transcript_chunk, 
    summarize_comment_batch, summarize_final_aggregation
)
from utils.formatters import format_summary
from utils.helpers import (
    with_retry, strip_banner, load_prompts, 
    chunk_text, chunk_list, comment_texts
)

# Configuration
BASE_DIR                = Path(__file__).parent
OUTPUT_DIR              = Path("output")
PROMPT_FILE             = BASE_DIR / "prompt.txt"
LLM_MODEL_DEFAULT       = "gpt-4o-mini"
MAX_TRANSCRIPT_CHUNK    = 10_000  # Chars per transcript chunk
MAX_COMMENT_BATCH       = 500     # Number of comments per batch
DEFAULT_WORKERS_OUTER   = 5
DEFAULT_WORKERS_INNER   = 3       # Parallelism within a single video's chunks

load_dotenv()
console = Console()

def process_single_video(v_dir, api_key, model, system_p, force, workers_inner):
    """Orchestrates hierarchical summarization for a single video."""
    video_id = v_dir.name
    t_path   = v_dir / "transcript.txt"
    c_path   = v_dir / "comments.json"
    ts_path  = v_dir / "_transcript_summary.txt" # Cache Stage 1
    s_path   = v_dir / "summary.txt"           # Final output

    if not t_path.exists() or not c_path.exists():
        return "skip", None

    if s_path.exists() and not force:
        return "skip", f"{video_id} — already summarized"

    try:
        # --- STAGE 1: Transcript Condensation ---
        raw_t = strip_banner(t_path.read_text(encoding="utf-8"))
        t_chunks = chunk_text(raw_t, MAX_TRANSCRIPT_CHUNK)
        
        print(f"  [{video_id}] Stage 1: Condensing transcript ({len(t_chunks)} chunks)...")
        chunk_results = []
        # We can process transcript chunks in parallel
        with ThreadPoolExecutor(max_workers=workers_inner) as exe:
            futures = {exe.submit(with_retry, summarize_transcript_chunk, ch, api_key, model, label=f"T-Chunk({video_id})"): ch for ch in t_chunks}
            for f in as_completed(futures):
                res, err = f.result()
                if err: return "fail", f"Stage 1 failed: {err}"
                chunk_results.append(res)
        
        transcript_summary = "\n\n".join(chunk_results)

        # --- STAGE 2: Chunked Comment Synthesis ---
        c_content = c_path.read_text(encoding="utf-8")
        c_data = json.loads(c_content)
        all_comments = c_data.get("comments", [])
        comment_batches = chunk_list(all_comments, MAX_COMMENT_BATCH)
        url = c_data.get("meta", {}).get("url", f"https://youtube.com/watch?v={video_id}")

        print(f"  [{video_id}] Stage 2: Processing comments ({len(all_comments)} total, {len(comment_batches)} batches)...")
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
        print(f"  [{video_id}] Stage 3: Generating master summary...")
        summaries_block = "\n\n".join(intermediate_summaries)
        final_summary, err = with_retry(
            summarize_final_aggregation, 
            transcript_summary, summaries_block, api_key, model, 
            label=f"Final({video_id})"
        )

        if err:
            return "fail", f"Stage 3 failed: {err}"

        raw_t = strip_banner(t_path.read_text(encoding="utf-8"))
        t_words = len(raw_t.split())
        formatted = format_summary(final_summary, video_id, url, model, t_words, len(all_comments))
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
    parser = argparse.ArgumentParser(description="Hierarchical Summary Generation for FLAP-T5.")
    parser.add_argument("--video", help="Specific Video ID to summarize")
    parser.add_argument("--force", action="store_true", help="Overwrite existing summaries")
    parser.add_argument("--model", default=None, help="LLM model to use")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS_OUTER, help="Videos to process in parallel")
    parser.add_argument("--workers-inner", type=int, default=DEFAULT_WORKERS_INNER, help="Parallelism within one video")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        print(f"[ERR] No API key found.")
        sys.exit(1)

    if args.model:
        model = args.model
    else:
        model = "gemini-1.5-flash" if os.getenv("GOOGLE_API_KEY") else LLM_MODEL_DEFAULT

    if args.video:
        v_dirs = [OUTPUT_DIR / args.video]
    else:
        v_dirs = [d for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]

    if not v_dirs:
        print(f"[ERR] No data found.")
        return

    console.print(f"[bold blue][HIERARCHICAL] Model: {model} | Outer: {args.workers} | Inner: {args.workers_inner}[/bold blue]\n")
    
    results = {"ok": 0, "skip": 0, "fail": 0}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
        expand=True,
        refresh_per_second=4
    ) as progress:
        task_id = progress.add_task("[cyan]Summarizing videos...", total=len(v_dirs))
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_video, v_dir, api_key, model, None, args.force, args.workers_inner): v_dir for v_dir in v_dirs}
            try:
                for future in as_completed(futures):
                    outcome, msg = future.result()
                    results[outcome] += 1
                    if outcome == "fail" and msg:
                        progress.console.print(f"  [red][FAIL] {msg}[/red]")
                    progress.advance(task_id)
            except KeyboardInterrupt:
                progress.console.print("\n[bold red][HALT] Summarization interrupted by user.[/bold red]")
                for f in futures:
                    f.cancel()
                pass

    console.print("-" * 30)
    console.print(f"[bold green][FINISHED][/bold green] Processed: [green]{results['ok']}[/green] | Skipped: [yellow]{results['skip']}[/yellow] | Failed: [red]{results['fail']}[/red]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[bold red][ERR] Summarization failed: {e}[/bold red]")
        sys.exit(1)
