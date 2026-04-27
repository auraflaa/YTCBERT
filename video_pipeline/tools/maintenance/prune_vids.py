"""
tools/maintenance/prune_vids.py
------------------------------
Thin-Data Purge Utility. Identifies and removes videos with incomplete extraction.
"""

import argparse
import sys
import shutil
import datetime
import os
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

# Local Imports
try:
    from utils.helpers import extract_video_id, resolve_data_path, rotate_backups
except ImportError:
    print(f"Error: Could not find 'utils' module. ROOT_DIR: {ROOT_DIR}")
    sys.exit(1)

console = Console()
VIDEO_FILE = resolve_data_path("video.txt")
OUTPUT_DIR = ROOT_DIR / "output"

def audit_video(v_id):
    """Checks if a video has actually yielded usable data."""
    v_dir = OUTPUT_DIR / v_id
    if not v_dir.exists():
        return False, "Output folder missing"
    
    t_path = v_dir / "transcript.txt"
    c_path = v_dir / "comments.json"
    
    if not t_path.exists() or t_path.stat().st_size < 10:
        return False, "Missing Transcript"
    if not c_path.exists() or c_path.stat().st_size < 10:
        return False, "Missing Comments"
    return True, "OK"

def main():
    parser = argparse.ArgumentParser(description="Prune 'dud' videos from video.txt.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not VIDEO_FILE.exists():
        console.print(f"[red][ERR] {VIDEO_FILE} not found.[/red]")
        return

    raw_lines = VIDEO_FILE.read_text(encoding="utf-8").splitlines()
    kept_lines = []
    removed_count = 0
    reasons = {}

    console.print(f"[bold blue]Auditing data in {OUTPUT_DIR}...[/bold blue]\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task("Auditing...", total=len(raw_lines))
        for line in raw_lines:
            v_id = extract_video_id(line.strip())
            if v_id and not line.strip().startswith("#"):
                is_valid, reason = audit_video(v_id)
                if is_valid:
                    kept_lines.append(line)
                else:
                    removed_count += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
            else:
                kept_lines.append(line)
            progress.advance(task)

    if removed_count == 0:
        console.print("\n[bold green]✅ No duds found.[/bold green]")
        return

    table = Table(title="Pruning Summary")
    table.add_column("Reason", style="red")
    table.add_column("Count", justify="right")
    for r, count in reasons.items():
        table.add_row(r, str(count))
    console.print(table)

    if not args.dry_run:
        bak_dir = ROOT_DIR / "backups"
        bak_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bak_path = bak_dir / f"video_prune_{ts}.bak"
        shutil.copy(VIDEO_FILE, bak_path)
        rotate_backups(bak_dir, max_keep=5)
        
        VIDEO_FILE.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        console.print(f"\n[bold green]Success![/bold green] Removed {removed_count} total entries.")
    else:
        console.print(f"\n[yellow][DRY RUN] Would have removed {removed_count} entries.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
