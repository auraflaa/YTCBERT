"""
Category-Aware Video Verification Utility.

Checks the status (Available, Private, Deleted) of URLs in video.txt.
Groups results by niche category and provides a distribution summary.
Uses the lightweight OEmbed API for fast, keyless validation.
Supports multi-threading and exports a `.verify_report.json` for fast link pruning.
"""
import argparse
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn

# Engine-relative imports
BASE_DIR = Path(__file__).parent
VIDEO_FILE = BASE_DIR / "video.txt"
DEFAULT_WORKERS = 10
console = Console()

try:
    from utils.helpers import extract_video_id
except ImportError:
    import re
    def extract_video_id(url):
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

def verify_single_video(url, category, line_no, seen_videos, lock):
    """Verifies existence and basic status of a single YouTube video with retries."""
    video_id = extract_video_id(url)
    if not video_id:
        return line_no, url, "INVALID", "Could not parse video ID", category
    
    with lock:
        if video_id in seen_videos:
            orig_lno = seen_videos[video_id]
            return line_no, video_id, "DUPLICATE", f"Same as line {orig_lno}", category
        seen_videos[video_id] = line_no

    max_retries = 5
    for attempt in range(max_retries):
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}"
            req = urllib.request.Request(oembed_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return line_no, video_id, "OK", data.get("title", "Unknown Title"), category
        except urllib.error.HTTPError as e:
            # Permanent errors: Don't retry
            if e.code == 404:
                return line_no, video_id, "NOT_FOUND", "Video does not exist (Deleted)", category
            if e.code in (401, 403):
                return line_no, video_id, "PRIVATE", "Video is private or restricted", category
            
            # Transient errors (like 500, 502, 503, 504, 429) - Retry
            if attempt < max_retries - 1:
                import time
                time.sleep(1) # Short wait before retry
                continue
            return line_no, video_id, "ERROR", f"HTTP {e.code}", category
        except Exception as e:
            # Network/Connection errors - Retry
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
            return line_no, video_id, "ERROR", str(e), category

def main():
    parser = argparse.ArgumentParser(description="Verify YouTube videos in video.txt using OEmbed.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers")
    args = parser.parse_args()

    if not VIDEO_FILE.exists():
        console.print(f"[red][ERR] {VIDEO_FILE} not found.[/red]")
        return

    lines = VIDEO_FILE.read_text(encoding="utf-8").splitlines()
    tasks = []
    current_cat = "Uncategorized"
    
    for i, line in enumerate(lines, 1):
        clean = line.strip()
        if clean.startswith("# Category:"):
            current_cat = clean.replace("# Category:", "").strip()
        elif clean and not clean.startswith("#"):
            tasks.append((clean, current_cat, i))

    if not tasks:
        console.print("[yellow]No video links found to verify.[/yellow]")
        return

    console.print(f"[bold blue]Verifying {len(tasks)} videos using {args.workers} workers...[/bold blue]")

    results = []
    seen_videos = {}
    lock = Lock()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        refresh_per_second=4
    ) as progress:
        task_id = progress.add_task("Verifying", total=len(tasks))
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(verify_single_video, t[0], t[1], t[2], seen_videos, lock) for t in tasks]
            try:
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.advance(task_id)
            except KeyboardInterrupt:
                progress.console.print("\n[bold red][HALT] Verification interrupted by user.[/bold red]")
                for f in futures:
                    f.cancel()
                # Proceed to show results for what was done

    # Sort results by line number
    results.sort(key=lambda x: x[0])

    table = Table(title="Video Verification Results")
    table.add_column("Line", justify="right")
    table.add_column("Category", style="magenta")
    table.add_column("ID/URL", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Info/Title")

    stats = {"OK": 0, "PRIVATE": 0, "NOT_FOUND": 0, "INVALID": 0, "DUPLICATE": 0, "ERROR": 0}
    
    for lno, vid, status, info, cat in results:
        stats[status] = stats.get(status, 0) + 1
        if status != "OK":
            table.add_row(str(lno), cat, vid, status, info)

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] " + " | ".join([f"{k}: {v}" for k, v in stats.items()]))

    # Save results for other tools to consume (Hidden in utils)
    report_file = BASE_DIR / "utils" / ".verify_report.json"
    report_data = {vid: status for lno, vid, status, info, cat in results}
    report_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    console.print(f"\n[dim]Report synced to utils/.verify_report.json for cleanup usage.[/dim]")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(f"\n[bold red][ERR] Verification failed: {e}[/bold red]")
        sys.exit(1)
