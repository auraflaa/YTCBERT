"""
tools/discovery/clean_video_links.py
------------------------------------
Category-Aware Video Link Cleaning Utility.
"""

import argparse
import sys
import json
import urllib.request
import urllib.error
import shutil
import datetime
import os
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Local Imports
try:
    from utils.helpers import extract_video_id, resolve_data_path, rotate_backups
except ImportError:
    print(f"Error: Could not find 'utils' module. ROOT_DIR: {ROOT_DIR}")
    sys.exit(1)

console = Console()
VIDEO_FILE = resolve_data_path("video.txt")

def is_video_available(v_id):
    """Returns True if the video is public and available, False otherwise. Retries on transient errors."""
    max_retries = 3 
    for attempt in range(max_retries):
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={v_id}"
            req = urllib.request.Request(oembed_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return False  
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            return True 
        except Exception:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            return True 

def clean_video_links(file_path, filter_private=False, apply_report=False, dry_run=False):
    """Main cleaning logic for deduplication and optional availability filtering."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red][ERR] File not found: {file_path}[/red]")
        return

    report_data = {}
    if apply_report:
        report_file = ROOT_DIR / "utils" / ".verify_report.json"
        if report_file.exists():
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
        else:
            console.print("[yellow][WARN] Verification report not found.[/yellow]")

    lines = path.read_text(encoding="utf-8").splitlines()
    unique_lines = []
    seen_ids = set()
    removed_count = 0
    private_count = 0
    current_cat_header = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        disable=not filter_private 
    ) as progress:
        task = progress.add_task("Cleaning Links...", total=len(lines))
        for line in lines:
            progress.advance(task)
            clean = line.strip()
            if not clean: continue
            if clean.startswith("# Category:"):
                current_cat_header = line
                continue
            if clean.startswith("#"):
                unique_lines.append(line)
                continue
            video_id = extract_video_id(clean)
            if not video_id:
                if current_cat_header:
                    unique_lines.append(current_cat_header)
                    current_cat_header = None
                unique_lines.append(line)
                continue
            if video_id in seen_ids:
                removed_count += 1
                continue
            if apply_report and video_id in report_data:
                if report_data[video_id] in ("PRIVATE", "NOT_FOUND", "DELETED", "RESTRICTED", "ERROR", "INVALID"):
                    private_count += 1
                    continue
            if filter_private and not apply_report:
                if not is_video_available(video_id):
                    private_count += 1
                    continue
            seen_ids.add(video_id)
            if current_cat_header:
                unique_lines.append(current_cat_header)
                current_cat_header = None
            unique_lines.append(line)

    if removed_count == 0 and private_count == 0:
        console.print("[green]No duplicates or private videos found.[/green]")
        return

    if dry_run:
        console.print(f"[yellow][DRY RUN] Would remove {removed_count} duplicates and {private_count} restricted videos.[/yellow]")
        return

    master_backup_dir = ROOT_DIR / "backups"
    master_backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = master_backup_dir / f"video_clean_{timestamp}.bak"
    shutil.copy(path, backup_path)
    rotate_backups(master_backup_dir, max_keep=5)
    
    path.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
    console.print(f"[bold green]Success![/bold green] Backup created: [dim]{backup_path.name}[/dim]")

def main():
    parser = argparse.ArgumentParser(description="Clean video.txt: Remove duplicates and filter private videos.")
    parser.add_argument("--filter-private", action="store_true")
    parser.add_argument("--apply-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    clean_video_links(VIDEO_FILE, args.filter_private, args.apply_report, args.dry_run)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red][HALT] Cleaning interrupted.[/bold red]")
        sys.exit(0)
