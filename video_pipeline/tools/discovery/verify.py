"""
tools/discovery/verify.py
-------------------------
Category-Aware Video Verification Utility.
Checks status (Available, Private, Deleted) of URLs in video.txt.
"""
import argparse
import sys
import json
import urllib.request
import urllib.error
import urllib.parse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from utils.helpers import extract_video_id, resolve_data_path

console = Console()
VIDEO_FILE = resolve_data_path("video.txt", base_dir=root_dir)

def verify_single_video(url, category, line_no):
    video_id = extract_video_id(url)
    if not video_id: return line_no, url, "INVALID", "Invalid ID", category
    max_retries = 3
    for attempt in range(max_retries):
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}"
            req = urllib.request.Request(oembed_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                return line_no, video_id, "OK", data.get("title", "Unknown"), category
        except urllib.error.HTTPError as e:
            if e.code == 404: return line_no, video_id, "NOT_FOUND", "Deleted", category
            if e.code in (401, 403): return line_no, video_id, "PRIVATE", "Private", category
            if attempt < max_retries - 1: continue
            return line_no, video_id, "ERROR", f"HTTP {e.code}", category
        except Exception:
            if attempt < max_retries - 1: continue
            return line_no, video_id, "ERROR", "Network Error", category

def main():
    parser = argparse.ArgumentParser(description="Verify videos in video.txt")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if not VIDEO_FILE.exists():
        console.print(f"[red]Error: {VIDEO_FILE} not found.[/red]")
        return

    lines = VIDEO_FILE.read_text(encoding="utf-8").splitlines()
    tasks, results, seen, current_cat = [], [], {}, "Uncategorized"
    
    for i, line in enumerate(lines, 1):
        clean = line.strip()
        if clean.startswith("# Category:"): current_cat = clean.replace("# Category:", "").strip()
        elif clean and not clean.startswith("#"):
            vid = extract_video_id(clean)
            if vid:
                if vid in seen: results.append((i, vid, "DUPLICATE", f"Line {seen[vid]}", current_cat))
                else: seen[vid] = i; tasks.append((clean, current_cat, i, vid))
            else: results.append((i, clean, "INVALID", "Parse Error", current_cat))

    if not tasks: return

    with Progress(TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=None), 
                  TaskProgressColumn(), TimeRemainingColumn(), console=console, expand=True) as progress:
        task_id = progress.add_task("Verifying...", total=len(tasks))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(verify_single_video, t[0], t[1], t[2]) for t in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                progress.advance(task_id)

    results.sort(key=lambda x: x[0])
    table = Table(title="Verification Results")
    table.add_column("Line", justify="right"); table.add_column("Category"); table.add_column("ID"); table.add_column("Status"); table.add_column("Info")
    
    for lno, vid, status, info, cat in results:
        if status != "OK": table.add_row(str(lno), cat, vid, status, info)
    console.print(table)

if __name__ == "__main__":
    main()
