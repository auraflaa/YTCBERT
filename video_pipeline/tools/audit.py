"""
tools/audit.py
--------------
Full audit of the output/ directory.
Reports completion status, duplicates, and engagement metrics.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich import box
from utils.helpers import resolve_data_path, parse_count

console = Console()

def _read_json(path: Path) -> dict:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

def _comment_count_from_file(c_path: Path) -> int:
    data = _read_json(c_path)
    return len(data.get("comments", []))

def _transcript_size(t_path: Path) -> int:
    try: return len(t_path.read_text(encoding="utf-8").strip())
    except Exception: return 0

def audit(output_dir: Path, min_comments: int = 0):
    if not output_dir.exists():
        console.print(f"[red]Error: {output_dir} not found.[/red]")
        return
    video_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if not video_dirs:
        console.print("[yellow]No data in output/.[/yellow]")
        return

    has_t, has_c, has_m, full, below = 0, 0, 0, 0, 0
    c_counts, t_sizes, v_ids, top = [], [], [], []
    broken = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console, transient=True) as progress:
        task = progress.add_task("Auditing...", total=len(video_dirs))
        for v_dir in video_dirs:
            vid = v_dir.name
            t_path, c_path, m_path = v_dir / "transcript.txt", v_dir / "comments.json", v_dir / "meta.json"
            v_ids.append(vid)
            t_ok = t_path.exists() and t_path.stat().st_size > 50
            c_ok = c_path.exists()
            if t_ok: has_t += 1; t_sizes.append(_transcript_size(t_path))
            if c_ok:
                has_c += 1; n = _comment_count_from_file(c_path); c_counts.append(n); top.append((n, vid))
                if min_comments > 0 and n < min_comments: below += 1
            if m_path.exists(): has_m += 1
            if t_ok and c_ok: full += 1
            if not t_ok and not c_ok: broken.append(vid)
            progress.advance(task)

    console.print(Panel.fit("[bold cyan]Dataset Audit Report[/bold cyan]"))
    t1 = Table(box=box.ROUNDED)
    t1.add_column("Metric"); t1.add_column("Value", justify="right")
    t1.add_row("Total Folders", f"{len(video_dirs):,}")
    t1.add_row("With Transcript", f"{has_t:,} ({has_t/len(video_dirs)*100:.1f}%)")
    t1.add_row("With Comments", f"{has_c:,} ({has_c/len(video_dirs)*100:.1f}%)")
    t1.add_row("Complete", f"{full:,} ({full/len(video_dirs)*100:.1f}%)")
    if min_comments > 0: t1.add_row("Below Min Comments", f"[yellow]{below}[/yellow]")
    t1.add_row("Broken", f"[red]{len(broken)}[/red]")
    console.print(t1)

    if top:
        top.sort(reverse=True)
        t2 = Table(title="Top 5 by Comments")
        t2.add_column("Video ID"); t2.add_column("Count", justify="right")
        for count, vid in top[:5]: t2.add_row(vid, str(count))
        console.print(t2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit output directory")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "output")
    parser.add_argument("--min-comments", type=parse_count, default=0)
    args = parser.parse_args()
    audit(args.output_dir, args.min_comments)
