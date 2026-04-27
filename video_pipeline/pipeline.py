"""
High-Performance YouTube Data Extractor (Step 1).

Manages the core data gathering flow:
1. Reads YouTube URLs from video_pipeline/video.txt.
2. Checks for fresh local data in output/ folders.
3. Concurrently fetches transcripts and comments with retry logic.
4. Saves data with atomic writes to prevent corruption.
5. Supports checkpointed resumption for comment downloads.
6. Gracefully handles Ctrl+C with a global StopSignal.
"""

import argparse
import json
import os
import sys
import time
import threading
import random
import shutil
import http.cookiejar
from datetime import datetime, timezone
from pathlib import Path
import langdetect
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.console import Console
console = Console()
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

# =============================================================================
# Core Exception types for graceful exits
# =============================================================================
class VideoUnavailable(Exception): pass

from utils.formatters import format_comments_json, format_transcript
from utils.throttle import global_throttle
from utils.helpers import (
    clean_err, extract_video_id, fmt_duration, 
    get_video_stats, needs_refresh, with_retry, 
    load_prompts, clean_comment_text, parse_count,
    resolve_data_path
)
from utils.stats import comment_texts, comments_meta, transcript_meta
from utils.proxies import rotator

# =============================================================================
# Configuration
# =============================================================================
CURR_DIR             = Path(__file__).resolve().parent
ROOT_DIR             = CURR_DIR.parent  # Project root

# Load project-wide environment
load_dotenv(ROOT_DIR / ".env")

# Local directories
DATA_DIR             = CURR_DIR / "data"
OUTPUT_DIR           = CURR_DIR / "output"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Path Resolution
VIDEO_LIST_FILE      = resolve_data_path("video.txt", base_dir=CURR_DIR)
FAILED_LIST_FILE     = resolve_data_path("FAILED.txt", base_dir=CURR_DIR)

MAX_COMMENTS         = 10000
REFRESH_AFTER_DAYS   = 30
YOUTUBE_API_KEY      = os.getenv("YOUTUBE_API_KEY", "")
COOKIES_PATH         = None

stop_event = threading.Event()
failed_lock = threading.Lock()  # Protect mutating the blacklist across threads

def unmark_failed(url: str):
    """Safely scrubs a recovered video from the permanent blacklist."""
    with failed_lock:
        try:
            p = Path(FAILED_LIST_FILE)
            if not p.exists(): return
            lines = p.read_text(encoding="utf-8").splitlines()
            if url in lines:
                lines.remove(url)
                if lines:
                    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                else:
                    p.write_text("", encoding="utf-8")
        except Exception: pass


# =============================================================================
# Data fetchers
# =============================================================================

# Initialize the Robust Transcriber Service
from utils.transcription import RobustTranscriber
transcriber = RobustTranscriber(OUTPUT_DIR)


_thread_local = threading.local()
def get_comment_downloader():
    """Returns a thread-local instance of the downloader with injected authentication."""
    if not hasattr(_thread_local, "downloader"):
        from youtube_comment_downloader import YoutubeCommentDownloader
        downloader = YoutubeCommentDownloader()
        
        # Inject cookies into the downloader's session
        if transcriber.auth.jar:
            downloader.session.cookies = transcriber.auth.jar
            
        _thread_local.downloader = downloader
    return _thread_local.downloader

def _fetch_comments_shared(video_id: str, max_comments: int, total_hint: int,
                           video_dir: Path, url_str: str) -> list[dict]:
    """
    Fetches comments for a video using the 3rd party scraper.
    Checks global stop_event for immediate halt.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    downloader = get_comment_downloader()
    comments, seen_ids = [], set()
    
    # Resumption Logic
    c_path = video_dir / "comments.json"
    if c_path.exists():
        try:
            old_data = json.loads(c_path.read_text(encoding="utf-8"))
            comments = old_data.get("comments", [])
            seen_ids = {c["cid"] for c in comments if "cid" in c}
        except Exception: pass
    
    start_time = time.time()
    new_count = 0
    checkpoint_batch = 500

    try:
        for item in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
            if stop_event.is_set():
                break

            cid = item.get("cid")
            if cid in seen_ids: continue
            
            text = clean_comment_text(item.get("text", ""))
            if not text: continue
            
            try:
                # Add language filter with fallback if library isn't working as expected
                if langdetect.detect(text) != 'en': continue
            except Exception: 
                # If langdetect fails (e.g., text too short or library issue), 
                # we keep the comment rather than skipping all of them!
                pass

            comments.append(item)
            seen_ids.add(cid)
            new_count += 1
            
            if new_count >= checkpoint_batch:
                _atomic_save_comments(c_path, comments, video_id, url_str)
                new_count = 0

            if max_comments > 0 and len(comments) >= max_comments:
                break
    except KeyboardInterrupt:
        stop_event.set()
        if new_count > 0:
            _atomic_save_comments(c_path, comments, video_id, url_str)
        raise
    
    if stop_event.is_set() and new_count > 0:
        _atomic_save_comments(c_path, comments, video_id, url_str)

    return comments


def _atomic_save_comments(path: Path, comments: list[dict], video_id: str, url: str):
    try:
        temp_path = path.with_suffix(".tmp")
        data = format_comments_json(comments, video_id, url)
        temp_path.write_text(data, encoding="utf-8")
        if path.exists(): path.unlink()
        temp_path.rename(path)
    except Exception as e:
        console.print(f"      [WARN] Failed to save checkpoint for {video_id}: {e}")


def _write_meta(video_dir: Path, video_id: str, url: str,
                transcript: str | None, comments: list[dict] | None,
                status: dict) -> None:
    meta = {
        "video_id":         video_id,
        "url":              url,
        "extracted_at":     datetime.now(timezone.utc).isoformat(),
        "status":           status,
        "transcript":       transcript_meta(transcript),
        "comments":         comments_meta(comments),
    }
    (video_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# Per-video processor
# =============================================================================

def process_video(url: str, idx: int, total: int, force: bool, refresh_days: int, max_comments: int, min_comments: int, browser_name: str | None, no_transcripts: bool, fast_fail: bool, progress, main_task) -> str:
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
    if stop_event.is_set():
        return "skip"

    # Anti-burst stagger: don't hit YouTube with 10 requests at the exact same ms
    time.sleep(random.uniform(0.1, 1.5))

    video_id = extract_video_id(url)
    if not video_id:
        if progress: progress.console.print(f"  [red]✖ Invalid URL[/red] {url}")
        if main_task is not None: progress.advance(main_task)
        return "fail"

    video_dir = OUTPUT_DIR / video_id
    if not force and not needs_refresh(video_dir, refresh_days, no_transcripts=no_transcripts):
        # Silent skip in loop to keep console clean, since we already did a pass in main()
        if main_task is not None: progress.advance(main_task)
        return "skip"
        
    if force and video_dir.exists():
        for item in video_dir.iterdir():
            if item.is_file():
                try: item.unlink()
                except Exception: pass

    if "/shorts/" in url.lower():
        if progress: progress.console.print(f"  [yellow]⏭ Skipped (Shorts)[/yellow] {url}")
        if main_task is not None: progress.advance(main_task)
        return "skip"

    # Stats / Meta
    yt = {}
    try:
        yt = get_video_stats(video_id, YOUTUBE_API_KEY)
    except Exception: pass
    
    if yt and yt.get("comments_disabled"):
        if progress: progress.console.print(f"  [yellow]⏭ Skipped (Comments Disabled)[/yellow] {url}")
        if main_task is not None: progress.advance(main_task)
        return "skip"

    if yt and 0 < yt.get("duration", 0) < 30:
        if progress: progress.console.print(f"  [yellow]⏭ Skipped (Duration < 30s)[/yellow] {url}")
        if main_task is not None: progress.advance(main_task)
        return "skip"

    n_total = yt.get("comment_count", 0) if yt else 0
    cap = max_comments if max_comments > 0 else n_total

    # Engagement Filter (Early Exit)
    if min_comments > 0 and n_total < min_comments:
        if progress: progress.console.print(f"  [yellow]⏭ Skipped (Engagement Low: {n_total}/{min_comments} comments)[/yellow] {url}")
        if main_task is not None: progress.advance(main_task)
        return "skip"

    # Creation happens ONLY after we are sure we want to process it
    video_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, str] = {}
    transcript: str | None = None
    comments: list[dict] | None = None
    any_error = False

    # Transcript
    t_path = video_dir / "transcript.txt"
    if no_transcripts:
        status["transcript"] = "skipped"
    elif not force and t_path.exists():
        status["transcript"] = "ok"
        transcript = t_path.read_text(encoding="utf-8")
    else:
        p_fetch = (VideoUnavailable,)
        transcript, err = with_retry(transcriber.get_transcript, video_id, url, attempts=1, label="Transcript", permanent_exceptions=p_fetch)
        if err:
            err_str = str(err).lower()
            if "too many requests" in err_str or "blocked" in err_str or "429" in err_str:
                global_throttle.report_429(long_block=(not fast_fail))
                err = f"{clean_err(err)} (Behavioral Block)"
            status["transcript"] = f"error: {err}"
            any_error = True
        else:
            # Data Integrity Layer: Prevent saving silently corrupt/empty payloads
            formatted_text = format_transcript(transcript, video_id, url)
            if len(formatted_text.strip()) < 50:
                global_throttle.report_success()
                status["transcript"] = "error: Silent Data Corruption (Valid fetch but empty extracted text)"
                any_error = True
            else:
                global_throttle.report_success()
                (video_dir / "transcript.txt").write_text(formatted_text, encoding="utf-8")
                status["transcript"] = "ok"

    # Comments
    c_path = video_dir / "comments.json"
    needs_comment_fetch = True
    if not force and c_path.exists():
        try:
            c_data = json.loads(c_path.read_text(encoding="utf-8"))
            comments_count = len(c_data.get("comments", []))
            if (max_comments > 0 and comments_count >= max_comments) or \
               (max_comments == 0 and n_total > 0 and comments_count >= (n_total * 0.95)):
                status["comments"] = "ok"
                needs_comment_fetch = False
        except Exception: pass

    if needs_comment_fetch and not stop_event.is_set():
        comments, err = with_retry(_fetch_comments_shared, video_id, max_comments, n_total, 
                                   video_dir, url)
        if err:
            status["comments"] = f"error: {err}"
            any_error = True
        else:
            # Post-fetch engagement check (for when API returns 0 or wrong stats)
            actual_count = len(comments)
            if min_comments > 0 and actual_count < min_comments:
                if progress: progress.console.print(f"  [yellow]⏭ Pruned (Actual comments {actual_count} < {min_comments})[/yellow] {url}")
                import shutil
                shutil.rmtree(video_dir, ignore_errors=True)
                if main_task is not None: progress.advance(main_task)
                return "skip"
                
            (video_dir / "comments.json").write_text(format_comments_json(comments, video_id, url), encoding="utf-8")
            status["comments"] = "ok"

    _write_meta(video_dir, video_id, url, transcript, comments, status)
    if main_task is not None: progress.advance(main_task)

    if any_error:
        import shutil
        shutil.rmtree(video_dir, ignore_errors=True)
        
        err_msg = status.get("transcript", "ok") if "error:" in status.get("transcript", "") else status.get("comments", "error")
        err_msg_lower = err_msg.lower()
        is_rate_limit = any(term in err_msg_lower for term in [
            "429", "rate limit", "too many requests", "sign in to confirm", "bot", "jsondecodeerror"
        ])
        
        # Log to blacklist ONLY if the video is actually broken, NOT if our IP/VPN is just temporarily banned
        if not is_rate_limit:
            with failed_lock:
                try:
                    # Check if already blacklisted to avoid duplicates
                    p = Path(FAILED_LIST_FILE)
                    existing = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
                    if url not in existing:
                        with open(FAILED_LIST_FILE, "a", encoding="utf-8") as f:
                            f.write(f"{url}\n")
                except Exception: pass
        else:
            if not stop_event.is_set():
                console.print(f"\n[bold red][FATAL] YouTube Rate Limit (429) active! Your IP is temporarily throttled.[/bold red] Halting pipeline to prevent falsely blacklisting healthy videos...")
                stop_event.set()
        
        if progress: progress.console.print(f"  [red]✖ Failed[/red] {url} -> {err_msg[:80]}...")
        return "fail"
        
    unmark_failed(url)
    if progress: progress.console.print(f"  [green]✔ Saved[/green] {url}")
    return "ok"


# =============================================================================
# CLI Main
# =============================================================================

def _load_urls(retry_failed: bool = False) -> list[str]:
    video_path = Path(VIDEO_LIST_FILE)
    failed_path = Path(FAILED_LIST_FILE)
    
    # Touch files if they don't exist natively
    if not video_path.exists():
        video_path.write_text("# Add YouTube URLs below (one per line):\n", encoding="utf-8")
    if not failed_path.exists():
        failed_path.write_text("", encoding="utf-8")

    try:
        lines = video_path.read_text(encoding="utf-8").splitlines()
        urls = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        if not urls:
            console.print(f"\n[bold yellow][WARN][/bold yellow] The file '{VIDEO_LIST_FILE}' is empty. Please paste some YouTube URLs into it, save, and run again.\n")
            sys.exit(0)
        
        if not retry_failed:
            try:
                failed_set = set(line.strip() for line in failed_path.read_text(encoding="utf-8").splitlines() if line.strip())
                urls = [u for u in urls if u not in failed_set]
            except Exception: pass
            
        return urls
    except Exception as e:
        console.print(f"[red][ERR] Could not load URLs from {VIDEO_LIST_FILE}: {e}[/red]")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="YouTube Pipeline Extractor")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-days", type=int, default=30)
    parser.add_argument("--max-comments", type=parse_count, default=10000)
    parser.add_argument("--workers", type=parse_count, default=4)
    parser.add_argument("--limit", type=parse_count, default=None, help="Process only the first N videos from the list")
    parser.add_argument("--min-comments", type=parse_count, default=0, help="Skip videos with fewer comments than this threshold")
    parser.add_argument("--retry-failed", action="store_true", help="Attempt to redownload videos tracked in failed_urls.txt")
    parser.add_argument("--cookies", type=str, default=None, help="Browser name to extract cookies from (e.g. chrome, edge, firefox, safari)")
    parser.add_argument("--no-transcripts", action="store_true", help="Skip transcript extraction entirely")
    parser.add_argument("--fast-fail", action="store_true", help="Don't wait for Hard Cooldowns (useful for dev/test)")
    args = parser.parse_args()
    transcriber.set_cookies(args.cookies)

    urls = _load_urls(retry_failed=args.retry_failed)
    if args.limit and args.limit > 0:
        urls = urls[:args.limit]
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = {"ok": 0, "skip": 0, "fail": 0}
    
    # INSTANT CACHE PRE-FILTER (Thorough)
    active_urls = []
    from utils.helpers import extract_video_id, needs_refresh
    for url in urls:
        vid = extract_video_id(url)
        if vid and not args.force:
            v_dir = OUTPUT_DIR / vid
            # Use same logic as process_video to decide if we skip
            if not needs_refresh(v_dir, args.refresh_days, no_transcripts=args.no_transcripts):
                results["skip"] += 1
                continue
        active_urls.append(url)
        
    if results["skip"] > 0:
        console.print(f"  [dim]⏭ Pre-filtered {results['skip']} fully cached videos from disk in 0.1s[/dim]")
        
    urls = active_urls

    def print_summary():
        console.print("\n" + "=" * 50)
        table = Table(title="Extraction Summary", box=None)
        table.add_column("Status", style="cyan")
        table.add_column("Count", justify="right", style="magenta")
        table.add_row("Success (OK)", str(results["ok"]))
        table.add_row("Skipped", str(results["skip"]))
        table.add_row("Failed", str(results["fail"]))
        console.print(table)
        console.print("=" * 50 + "\n")

    if not urls:
        console.print("\n[bold green]All targeted videos are already entirely cached! Nothing left to process![/bold green]")
        return

    console.print(f"[bold blue][PIPELINE][/bold blue] Processing {len(urls)} active videos with {args.workers} workers...")

    # Compact Master Progress
    with Progress(
        TextColumn("[bold blue]{task.fields[status]}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
        refresh_per_second=2 # Slow down refresh to prevent jitter on Windows
    ) as progress:
        status_text = f"OK: {results['ok']} | Skip: {results['skip']} | Fail: {results['fail']}"
        main_task = progress.add_task("Queue", total=len(urls), status=status_text)
        
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = {
            executor.submit(process_video, url, i, len(urls), args.force, 
                           args.refresh_days, args.max_comments, args.min_comments,
                           args.cookies, args.no_transcripts, args.fast_fail, 
                           progress, main_task): url 
            for i, url in enumerate(urls, 1)
        }
        
        try:
            while futures:
                done, futures_list = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                for f in done:
                    outcome = f.result()
                    results[outcome] += 1
                    
                    # Update live status bar
                    status_text = f"OK: {results['ok']} | Skip: {results['skip']} | Fail: {results['fail']}"
                    progress.update(main_task, status=status_text)
                    
                    del futures[f]
                
                if stop_event.is_set():
                    break
        except KeyboardInterrupt:
            stop_event.set()
            console.print("\n\n[bold red][HALT] Interrupt detected! Shutting down...[/bold red]")
            executor.shutdown(wait=False, cancel_futures=True)
            print_summary()
            os._exit(1)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    print_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        console.print("\n\n[bold red][HALT] Pipeline interrupted by user.[/bold red]")
        os._exit(0)
