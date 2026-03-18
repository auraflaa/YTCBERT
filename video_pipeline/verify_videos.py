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
import urllib.parse
import os
from dotenv import load_dotenv
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
load_dotenv()

try:
    from utils.helpers import extract_video_id
except ImportError:
    import re
    def extract_video_id(url):
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

def verify_single_video(url, category, line_no):
    """Verifies existence and basic status of a single YouTube video with retries."""
    video_id = extract_video_id(url)
    if not video_id:
        return line_no, url, "INVALID", "Could not parse video ID", category

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
def parse_yt_duration(duration_str):
    import re
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str or "")
    if not match: return 0
    h, m, s = match.groups()
    return int(h or 0) * 3600 + int(m or 0) * 60 + int(s or 0)

def verify_api_batch(batch, api_key):
    """Verifies a batch of up to 50 videos via YouTube API."""
    batch_results = []
    missing_or_error = []
    
    vid_map = {t[3]: t for t in batch} # v_id -> (url, cat, lno, vid_id)
    ids_param = ",".join(vid_map.keys())
    
    try:
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,status,contentDetails,statistics&id={ids_param}&key={api_key}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            
        found_ids = set()
        for item in data.get("items", []):
            v_id = item["id"]
            found_ids.add(v_id)
            t = vid_map[v_id]
            
            status_block = item.get("status", {})
            dur_str = item.get("contentDetails", {}).get("duration", "PT0S")
            dur_sec = parse_yt_duration(dur_str)
            comment_count = int(item.get("statistics", {}).get("commentCount", 0))

            if status_block.get("privacyStatus") not in ("public", "unlisted"):
                 batch_results.append((t[2], v_id, "PRIVATE", "Video is not public", t[1]))
            elif status_block.get("embeddable") is False:
                 batch_results.append((t[2], v_id, "RESTRICTED", "Embeds disabled", t[1]))
            elif dur_sec < 30:
                 batch_results.append((t[2], v_id, "INVALID", f"Too short ({dur_sec}s)", t[1]))
            elif comment_count == 0:
                 batch_results.append((t[2], v_id, "INVALID", "No comments", t[1]))
            else:
                 batch_results.append((t[2], v_id, "OK", item["snippet"].get("title", ""), t[1]))
        
        missing_ids = set(vid_map.keys()) - found_ids
        for m_id in missing_ids:
            missing_or_error.append(vid_map[m_id])
            
    except Exception:
        missing_or_error.extend(batch)
        
    return batch_results, missing_or_error

def main():
    parser = argparse.ArgumentParser(description="Verify YouTube videos in video.txt.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers for oEmbed fallback")
    args = parser.parse_args()

    if not VIDEO_FILE.exists():
        console.print(f"[red][ERR] {VIDEO_FILE} not found.[/red]")
        return

    lines = VIDEO_FILE.read_text(encoding="utf-8").splitlines()
    tasks = []
    current_cat = "Uncategorized"
    
    seen_videos = {}
    lock = Lock()
    results = []
    
    for i, line in enumerate(lines, 1):
        clean = line.strip()
        if clean.startswith("# Category:"):
            current_cat = clean.replace("# Category:", "").strip()
        elif clean and not clean.startswith("#"):
            vid_id = extract_video_id(clean)
            if vid_id:
                if vid_id in seen_videos:
                    results.append((i, vid_id, "DUPLICATE", f"Same as line {seen_videos[vid_id]}", current_cat))
                else:
                    seen_videos[vid_id] = i
                    tasks.append((clean, current_cat, i, vid_id))
            else:
                results.append((i, clean, "INVALID", "Could not parse video ID", current_cat))

    if not tasks:
        console.print("[yellow]No video links found to verify.[/yellow]")
        return

    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        console.print(f"[bold blue]Executing Verification Engine...[/bold blue]")
        batch_size = 50
    else:
        console.print(f"[bold blue]Verifying {len(tasks)} videos via oEmbed using {args.workers} workers...[/bold blue]")
        batch_size = 1

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
        
        if api_key:
            # BATCHED API APPROACH (MULTI-THREADED)
            batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
            missing_or_error_tasks = []
            
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(verify_api_batch, b, api_key) for b in batches]
                try:
                    for future in as_completed(futures):
                        batch_res, batch_missing = future.result()
                        results.extend(batch_res)
                        missing_or_error_tasks.extend(batch_missing)
                        # Advance progress by the number of successes in this batch
                        progress.advance(task_id, advance=len(batch_res))
                except KeyboardInterrupt:
                    progress.console.print("\n[bold red][HALT] Verification interrupted by user.[/bold red]")
                    for f in futures:
                        f.cancel()

            # FALLBACK: Double-check any missing items using original multi-threaded oEmbed
            if missing_or_error_tasks:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(verify_single_video, t[0], t[1], t[2]) for t in missing_or_error_tasks]
                    for future in as_completed(futures):
                        results.append(future.result())
                        progress.advance(task_id, advance=1)
                        
        else:
            # ORIGINAL OEMBED THREADED APPROACH (No API Key)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(verify_single_video, t[0], t[1], t[2]) for t in tasks]
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                        progress.advance(task_id)
                except KeyboardInterrupt:
                    progress.console.print("\n[bold red][HALT] Verification interrupted by user.[/bold red]")
                    for f in futures:
                        f.cancel()

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
