"""
Video Verification Utility.

Checks a list of YouTube URLs for availability (Private/Not Found) and 
detects duplicates within the source file. Uses the YouTube OEmbed API 
for lightweight status checks without requiring a full Data API key.
"""
import argparse
import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from utils.helpers import extract_video_id, get_video_stats

# Configuration
BASE_DIR = Path(__file__).parent
VIDEO_FILE = BASE_DIR / "video.txt"
DEFAULT_WORKERS = 10

load_dotenv()
console = Console()

def verify_single_video(url, line_no, api_key, seen_videos):
    """Verifies existence and basic status of a single YouTube video."""
    video_id = extract_video_id(url)
    if not video_id:
        return line_no, url, "INVALID", "Could not parse video ID"
    
    # Check for duplicates
    if video_id in seen_videos:
        orig_lno = seen_videos[video_id]
        return line_no, video_id, "DUPLICATE", f"Same as line {orig_lno}"
    
    seen_videos[video_id] = line_no

    # Try granular status via OEmbed (No API key needed, very specific)
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}"
        req = urllib.request.Request(oembed_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return line_no, video_id, "OK", data.get("title", "Unknown Title")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return line_no, video_id, "NOT_FOUND", "Video does not exist (Deleted)"
        if e.code in (401, 403):
            return line_no, video_id, "PRIVATE", "Video is private or restricted"
        return line_no, video_id, "ERROR", f"HTTP {e.code}"
    except Exception as e:
        # Fallback to standard API if OEmbed fails for some reason
        stats = get_video_stats(video_id, api_key)
        if not stats:
            return line_no, video_id, "MISSING", "Not found or private"
        return line_no, video_id, "OK", stats.get("title", "Unknown Title")

def main():
    parser = argparse.ArgumentParser(description="Verify existence of videos in video.txt.")
    parser.add_argument("--file", default=VIDEO_FILE, help=f"Path to video list (default: {VIDEO_FILE})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        console.print("[yellow][WARN] YOUTUBE_API_KEY not found. Using OEmbed fallback only.[/yellow]")

    path = Path(args.file)
    if not path.exists():
        console.print(f"[red][ERR] File not found: {args.file}[/red]")
        return

    # Load URLs with line numbers
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    tasks_to_run = []
    for i, line in enumerate(raw_lines, 1):
        clean = line.strip()
        if clean and not clean.startswith("#"):
            tasks_to_run.append((clean, i))

    if not tasks_to_run:
        console.print(f"[yellow][WARN] {args.file} has no valid URLs.[/yellow]")
        return

    console.print(f"[bold blue]Verifying {len(tasks_to_run)} videos...[/bold blue]")

    results = []
    seen_videos = {} # ID -> First Line No
    
    try:
        # We process sequentially for duplicate detection logic to be simple, 
        # or use a thread-safe dict. Let's use a thread-safe approach.
        from threading import Lock
        lock = Lock()
        
        shared_seen = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Verifying", total=len(tasks_to_run))
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # We can't easily check duplicates in parallel WITHOUT a lock or pre-processing
                # Let's pre-process IDs to handle duplicates instantly
                futures = []
                for url, lno in tasks_to_run:
                    futures.append(executor.submit(verify_single_video, url, lno, api_key, shared_seen))
                
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.advance(task)
                    
    except KeyboardInterrupt:
        console.print("\n[bold red][HALT] Interrupted by user. Showing partial results...[/bold red]")

    # Sort results by line number
    results.sort(key=lambda x: x[0])

    table = Table(title="Video Verification Results")
    table.add_column("Line", style="dim", justify="right")
    table.add_column("ID/URL", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Info/Title", style="white")

    counts = {"OK": 0, "NOT_FOUND": 0, "PRIVATE": 0, "INVALID": 0, "DUPLICATE": 0, "ERROR": 0, "MISSING": 0}
    for lno, vid, status, info in results:
        counts[status] = counts.get(status, 0) + 1
        
        # Color mapping
        color = "green" if status == "OK" else "red" if status in ("NOT_FOUND", "MISSING") else "yellow"
        if status == "PRIVATE": color = "magenta"
        if status == "DUPLICATE": color = "blue"
        
        if status != "OK":
            table.add_row(str(lno), vid, f"[{color}]{status}[/{color}]", info)

    if table.row_count > 0:
        console.print(table)
    else:
        console.print("\n[bold green]✅ All videos in the list verified successfully![/bold green]")

    console.print("-" * 30)
    summary_parts = [
        f"[green]Valid: {counts.get('OK',0)}[/green]",
        f"[red]Not Found: {counts.get('NOT_FOUND',0)}[/red]",
        f"[magenta]Private: {counts.get('PRIVATE',0)}[/magenta]",
        f"[yellow]Invalid: {counts.get('INVALID',0)}[/yellow]",
        f"[blue]Duplicates: {counts.get('DUPLICATE',0)}[/blue]"
    ]
    console.print(" | ".join(summary_parts))

if __name__ == "__main__":
    main()
