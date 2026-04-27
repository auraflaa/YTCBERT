"""
tools/precheck.py
-----------------
Scans input video.txt and provides metadata assessment before extraction.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.panel import Panel
from dotenv import load_dotenv
from utils.helpers import extract_video_id, get_video_stats_batch, fmt_duration, resolve_data_path

load_dotenv()
console = Console()

def precheck(video_file: Path, api_key: str):
    if not video_file.exists():
        console.print(f"[red]Error: {video_file} not found.[/red]")
        return
    lines = video_file.read_text(encoding="utf-8").splitlines()
    u_ids, id_to_cat, cur_cat = [], {}, "Uncategorized"
    
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith("# Category:"): cur_cat = line.split(":", 1)[1].strip(); continue
        if line.startswith("#"): continue
        vid = extract_video_id(line)
        if vid and vid not in id_to_cat: u_ids.append(vid); id_to_cat[vid] = cur_cat

    if not u_ids:
        console.print("[yellow]No videos found.[/yellow]")
        return

    console.print(f"\n[bold blue]Assessing {len(u_ids)} videos...[/bold blue]\n")
    all_stats = {}
    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console) as progress:
        task = progress.add_task("Fetching meta...", total=len(u_ids))
        for i in range(0, len(u_ids), 50):
            batch = u_ids[i : i + 50]
            try: all_stats.update(get_video_stats_batch(batch, api_key))
            except Exception: pass
            progress.advance(task, len(batch))

    total_d, total_c = 0, 0
    for stats in all_stats.values():
        total_d += stats.get("duration", 0)
        total_c += stats.get("comment_count", 0)

    console.print(Panel.fit(f"[bold green]Verified: {len(all_stats)}/{len(u_ids)}[/bold green]\n"
                            f"Est. Duration: [yellow]{fmt_duration(total_d)}[/yellow]\n"
                            f"Est. Comments: [yellow]{total_c:,}[/yellow]", border_style="blue"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-check source list")
    parser.add_argument("--file", type=str, default="video.txt")
    args = parser.parse_args()
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    precheck(resolve_data_path(args.file, base_dir=root_dir), api_key)
