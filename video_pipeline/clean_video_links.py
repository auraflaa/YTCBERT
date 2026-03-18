"""
Category-Aware Video Link Cleaning Utility.

Cleans video_pipeline/video.txt by:
1. Removing duplicate URLs across the entire list.
2. Deduplicating niche category headers.
3. Pruning empty categories that no longer contain links.
4. Optional: Filtering out private/deleted videos (via --filter-private or --apply-report).
"""
import argparse
import sys
import json
import urllib.request
import urllib.error
import shutil
import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Engine-relative imports
BASE_DIR = Path(__file__).parent
VIDEO_FILE = BASE_DIR / "video.txt"
console = Console()

try:
    from utils.helpers import extract_video_id
except ImportError:
    import re
    def extract_video_id(url):
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

def is_video_available(v_id):
    """Returns True if the video is public and available, False otherwise. Retries on transient errors."""
    max_retries = 3 # Use slightly fewer retries for cleaning to stay faster
    for attempt in range(max_retries):
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={v_id}"
            req = urllib.request.Request(oembed_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return False  # Permanent states
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            return True # Fallback to keeping it if we're unsure
        except Exception:
            if attempt < max_retries - 1:
                import time
                time.sleep(0.5)
                continue
            return True # Assume OK on generic connection errors to stay safe

def clean_video_links(file_path, filter_private=False, apply_report=False, dry_run=False):
    """Main cleaning logic for deduplication and optional availability filtering."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red][ERR] File not found: {file_path}[/red]")
        return

    report_data = {}
    if apply_report:
        report_file = BASE_DIR / "utils" / ".verify_report.json"
        if report_file.exists():
            report_data = json.loads(report_file.read_text(encoding="utf-8"))
            console.print(f"[dim]Loaded verification report from utils/.verify_report.json[/dim]")
        else:
            console.print("[yellow][WARN] Verification report not found. Run verify_videos.py first.[/yellow]")

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
        disable=not filter_private # Only show progress if we are doing network checks
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
                
            # 1. Duplicate check
            if video_id in seen_ids:
                removed_count += 1
                continue
            
            # 2. Check cached report (Fast)
            if apply_report and video_id in report_data:
                status = report_data[video_id]
                if status in ("PRIVATE", "NOT_FOUND", "DELETED", "RESTRICTED", "ERROR", "INVALID"):
                    private_count += 1
                    continue

            # 3. Availability filter (On-the-fly, Slower)
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
        console.print("[green]No duplicates or private videos found. The list is clean.[/green]")
        return

    if dry_run:
        console.print(f"[yellow][DRY RUN] Would remove {removed_count} duplicates and {private_count} restricted videos.[/yellow]")
        return

    # Create backup in the master root backup folder
    master_backup_dir = BASE_DIR.parent / "backups"
    master_backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = master_backup_dir / f"video_clean_{timestamp}.bak"
    
    shutil.copy(path, backup_path)
    path.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
    
    console.print(f"[bold green]Success![/bold green] Results:")
    console.print(f" - Backup created: [dim]{backup_path.name}[/dim]")
    if removed_count: console.print(f" - Removed {removed_count} duplicate link(s).")
    if private_count: console.print(f" - Filtered {private_count} private/restricted video(s).")

def main():
    parser = argparse.ArgumentParser(description="Clean video.txt: Remove duplicates and filter private videos.")
    parser.add_argument("--filter-private", action="store_true", help="Verify and remove restricted/private videos (Live network check)")
    parser.add_argument("--apply-report", action="store_true", help="Use results from verify_report.json to purge restricted videos (Instant)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without changing files")
    args = parser.parse_args()

    clean_video_links(VIDEO_FILE, filter_private=args.filter_private, apply_report=args.apply_report, dry_run=args.dry_run)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red][HALT] Cleaning interrupted.[/bold red]")
        sys.exit(0)
