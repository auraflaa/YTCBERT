"""
tools/maintenance/check_progress.py
----------------------------------
Full audit of the Video Pipeline output directory.
"""

import argparse
import json
import statistics
import sys
import os
from collections import Counter
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import box

# Local Imports
try:
    from utils.helpers import resolve_data_path, parse_count
except ImportError:
    print(f"Error: Could not find 'utils' module. ROOT_DIR: {ROOT_DIR}")
    sys.exit(1)

console = Console()

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _comment_count_from_file(c_path: Path) -> int:
    if not c_path.exists():
        return 0
    data = _read_json(c_path)
    comments = data.get("comments", [])
    return len(comments)

def _transcript_size(t_path: Path) -> int:
    if not t_path.exists():
        return 0
    try:
        return len(t_path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0

def audit(output_dir: Path, min_comments: int = 0):
    if not output_dir.exists():
        console.print(f"[red]Error: directory '{output_dir}' not found.[/red]")
        return

    video_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    total = len(video_dirs)

    if total == 0:
        console.print("[yellow]No video folders found in output/.[/yellow]")
        return

    has_transcript   = 0
    has_comments     = 0
    has_meta         = 0
    fully_complete   = 0   
    comments_only    = 0   
    broken           = []  

    comment_counts   = []
    transcript_sizes = []
    video_ids        = []
    top_videos       = []  
    below_threshold  = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Auditing folders...", total=total)
        
        for v_dir in video_dirs:
            vid          = v_dir.name
            t_path       = v_dir / "transcript.txt"
            c_path       = v_dir / "comments.json"
            m_path       = v_dir / "meta.json"
            video_ids.append(vid)

            t_ok = t_path.exists() and t_path.stat().st_size > 50
            c_ok = c_path.exists()
            m_ok = m_path.exists()

            if t_ok:
                has_transcript += 1
                transcript_sizes.append(_transcript_size(t_path))
            if c_ok:
                has_comments += 1
                n = _comment_count_from_file(c_path)
                comment_counts.append(n)
                top_videos.append((n, vid))
                if min_comments > 0 and n < min_comments:
                    below_threshold += 1
            if m_ok:
                has_meta += 1
            if t_ok and c_ok:
                fully_complete += 1
            elif c_ok and not t_ok:
                comments_only += 1
            elif not t_ok and not c_ok:
                broken.append(vid)
            progress.advance(task)

    id_counts   = Counter(video_ids)
    duplicates  = {vid: cnt for vid, cnt in id_counts.items() if cnt > 1}
    n_dupes     = sum(cnt - 1 for cnt in duplicates.values())
    top_videos.sort(reverse=True)

    console.print()
    console.print(Panel.fit("[bold cyan]YouTube Dataset Audit Report[/bold cyan]", border_style="bright_blue"))
    t1 = Table(title="📦 Overview", box=box.ROUNDED, header_style="bold blue")
    t1.add_column("Metric", style="cyan", no_wrap=True)
    t1.add_column("Value", style="bright_white", justify="right")
    t1.add_row("Total Video Folders", f"{total:,}")
    t1.add_row("Has Transcript", f"{has_transcript:,} ({has_transcript/total*100:.1f}%)")
    t1.add_row("Has Comments", f"{has_comments:,} ({has_comments/total*100:.1f}%)")
    t1.add_row("Has Meta JSON", f"{has_meta:,} ({has_meta/total*100:.1f}%)")
    t1.add_row("Fully Complete", f"{fully_complete:,} ({fully_complete/total*100:.1f}%)")
    t1.add_row("Comments Only", f"{comments_only:,}")
    t1.add_row("Below Engagement", f"[yellow]{below_threshold}[/yellow]" if min_comments > 0 else "N/A")
    t1.add_row("Broken Folders", f"[red]{len(broken)}[/red]")
    t1.add_row("Duplicates", f"[{'red' if n_dupes else 'green'}]{n_dupes}[/]")
    console.print(t1)

    if comment_counts:
        c_avg = statistics.mean(comment_counts)
        c_med = statistics.median(comment_counts)
        t2 = Table(title="💬 Comment Stats", box=box.ROUNDED, header_style="bold blue")
        t2.add_column("Metric", style="cyan")
        t2.add_column("Value", justify="right")
        t2.add_row("Total Stored", f"{sum(comment_counts):,}")
        t2.add_row("Average", f"{c_avg:,.1f}")
        t2.add_row("Median", f"{c_med:,.0f}")
        console.print(t2)

    if top_videos:
        t4 = Table(title="🏆 Top 10 by Comments", box=box.ROUNDED, header_style="bold blue")
        t4.add_column("Rank", style="dim", justify="right")
        t4.add_column("Video ID", style="cyan")
        t4.add_column("Comments", justify="right")
        for rank, (count, vid) in enumerate(top_videos[:10], 1):
            t4.add_row(str(rank), vid, f"{count:,}")
        console.print(t4)

    health_pct = (has_comments / total * 100) if total else 0
    color = "green" if health_pct >= 90 else "yellow" if health_pct >= 70 else "red"
    console.print(Panel(f"[{color}]Dataset Health: {health_pct:.1f}%[/{color}] ({has_comments}/{total} with comments)", border_style=color))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit the pipeline output directory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "output",
                        help="Path to the output directory (default: ./output)")
    parser.add_argument("--min-comments", type=parse_count, default=0)
    args = parser.parse_args()
    audit(args.output_dir, args.min_comments)
