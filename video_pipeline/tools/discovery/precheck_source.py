"""
tools/discovery/precheck_source.py
----------------------------------
Scans the input video list and provides a summary.
"""

import os
import sys
import argparse
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel

# Local imports
try:
    from utils.helpers import extract_video_id, get_video_stats_batch, fmt_duration, resolve_data_path
except ImportError:
    print(f"Error: Could not find 'utils' module. ROOT_DIR: {ROOT_DIR}")
    sys.exit(1)

console = Console()

def precheck(video_file: Path, api_key: str):
    if not video_file.exists():
        console.print(f"[red]Error: {video_file} not found.[/red]")
        return

    lines = video_file.read_text(encoding="utf-8").splitlines()
    unique_ids = []
    id_to_cat = {}
    current_cat = "Uncategorized"
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("# Category:"):
            current_cat = line.split(":", 1)[1].strip()
            continue
        if line.startswith("#"): continue
        
        vid = extract_video_id(line)
        if vid:
            if vid not in id_to_cat:
                unique_ids.append(vid)
                id_to_cat[vid] = current_cat

    total_count = len(unique_ids)
    if total_count == 0:
        console.print("[yellow]No video IDs found in the file.[/yellow]")
        return

    console.print(f"\n[bold blue]Pre-checking {total_count} unique videos...[/bold blue]\n")
    all_stats = {}
    batch_size = 50
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Fetching metadata...", total=total_count)
        for i in range(0, total_count, batch_size):
            batch = unique_ids[i : i + batch_size]
            try:
                res = get_video_stats_batch(batch, api_key)
                all_stats.update(res)
            except Exception: pass
            progress.advance(task, len(batch))

    # --- Analysis & Reporting ---
    cat_counts = {}
    total_duration = 0
    total_comments = 0
    found_count = len(all_stats)

    for vid, stats in all_stats.items():
        cat = id_to_cat.get(vid, "Uncategorized")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        total_duration += stats.get("duration", 0)
        total_comments += stats.get("comment_count", 0)

    console.print(Panel.fit(
        f"[bold green]Source Assessment Complete[/bold green]\n"
        f"Unique Videos: {total_count:,}\n"
        f"Verified Meta: {found_count:,}\n"
        f"Est. Duration: {fmt_duration(total_duration)}",
        border_style="blue"
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the video list.")
    parser.add_argument("--file", type=str, default="video.txt")
    args = parser.parse_args()

    v_path = resolve_data_path(args.file)
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    precheck(v_path, api_key)
