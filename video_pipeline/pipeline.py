"""
High-Performance YouTube Data Extractor (Step 1).

Manages the core data gathering flow:
1. Reads YouTube URLs from video_pipeline/video.txt.
2. Checks for fresh local data in output/ folders.
3. Concurrently fetches transcripts and comments with retry logic.
4. Saves data with atomic writes to prevent corruption.
5. Supports checkpointed resumption for comment downloads.
6. Gracefully handles Ctrl+C to save in-flight work.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.console import Console
console = Console()
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
import langdetect
from langdetect.lang_detect_exception import LangDetectException

from utils.formatters import format_comments_json, format_transcript
from utils.helpers import (
    clean_err, extract_video_id, fmt_duration, 
    get_video_stats, needs_refresh, with_retry, 
    load_prompts, clean_comment_text, parse_count
)
from utils.stats import comment_texts, comments_meta, transcript_meta

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR             = Path(__file__).parent
VIDEO_LIST_FILE      = BASE_DIR / "video.txt"
OUTPUT_DIR           = Path("output")  # Still relative to project root execution
MAX_COMMENTS         = 10000    # Cap at 10k comments (better for BERT analysis)
REFRESH_AFTER_DAYS   = 30
RETRY_ATTEMPTS       = 3
RETRY_BACKOFF_BASE   = 2        # seconds; doubles each retry
LIVE_PRINT_EVERY     = 50       # print progress every N comments

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")


# =============================================================================
# Data fetchers
# =============================================================================


def _fetch_transcript(video_id: str) -> str:
    # Explicitly request English transcript (manual or auto-generated)
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    return "\n".join(s.text for s in fetched)


def _fetch_comments(video_id: str, max_comments: int = 0,
                    total_hint: int = 0, video_dir: Path = None, url_str: str = "") -> list[dict]:
    """
    Fetches comments with a live progress bar. 
    Supports checkpointed resumption if video_dir is provided.
    """
    url        = f"https://www.youtube.com/watch?v={video_id}"
    downloader = YoutubeCommentDownloader()
    
    comments: list[dict] = []
    seen_ids: set[str]   = set()
    
    # --- Resumption Logic ---
    c_path = video_dir / "comments.json" if video_dir else None
    if c_path and c_path.exists():
        try:
            old_data = json.loads(c_path.read_text(encoding="utf-8"))
            comments = old_data.get("comments", [])
            seen_ids = {c["cid"] for c in comments if "cid" in c}
            if comments:
                print(f"      [RESUME] Found {len(comments)} existing comments. Continuing...")
        except Exception:
            pass # Start fresh if corrupted
    
    cap = max_comments if max_comments > 0 else total_hint
    checkpoint_batch = 500
    new_count = 0

    with Progress(
        TextColumn("  [cyan]Fetching..."),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        # rate is custom field
        TextColumn("•"),
        TextColumn("[green]{task.fields[rate]} c/s"),
        TextColumn("•"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("download", total=cap if cap > 0 else None, rate="0.0")
        # Initialize progress for resumed data
        progress.update(task, completed=len(comments))
        
        start_time = time.time()
        
        try:
            for item in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
                cid = item.get("cid")
                if cid in seen_ids:
                    continue
                    
                text = clean_comment_text(item.get("text", ""))
                if not text: continue
                
                item["text"] = text

                # Minimal language sanity
                try:
                    if langdetect.detect(text) != 'en': continue
                except Exception: continue

                comments.append(item)
                seen_ids.add(cid)
                new_count += 1
                
                if video_dir and new_count >= checkpoint_batch:
                    _atomic_save_comments(c_path, comments, video_id, url_str)
                    new_count = 0
                
                elapsed = time.time() - start_time
                n = len(comments)
                rate_val = (n - (len(comments) - new_count)) / elapsed if elapsed > 0 else 0.0
                progress.update(task, completed=n, rate=f"{rate_val:.1f}")

                if max_comments > 0 and len(comments) >= max_comments:
                    break
        except KeyboardInterrupt:
            if video_dir and new_count > 0:
                print(f"\n      [HALT] Interrupted. Flushing {new_count} new comments to disk...")
                _atomic_save_comments(c_path, comments, video_id, url_str)
            raise # Re-raise to be caught by the thread pool or main logic

    return comments

def _atomic_save_comments(path: Path, comments: list[dict], video_id: str, url: str):
    """Saves comments to a temp file and renames it to prevent corruption."""
    try:
        temp_path = path.with_suffix(".tmp")
        data = format_comments_json(comments, video_id, url)
        temp_path.write_text(data, encoding="utf-8")
        if path.exists():
            path.unlink()
        temp_path.rename(path)
    except Exception as e:
        print(f"      [WARN] Failed to save checkpoint: {e}")




def _short_num(n: int) -> str:
    """Formats large numbers (e.g., 1500 -> 1.5k)."""
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}k"
    return str(n)


# =============================================================================
# Meta writer
# =============================================================================

def _write_meta(video_dir: Path, video_id: str, url: str,
                transcript: str | None, comments: list[dict] | None,
                status: dict) -> None:
    meta = {
        "video_id":         video_id,
        "url":              url,
        "extracted_at":     datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0",
        "status":           status,
        "transcript":       transcript_meta(transcript),
        "comments":         comments_meta(comments),
    }
    (video_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# =============================================================================
# Per-video processor
# =============================================================================

def process_video(url: str, idx: int, total: int, force: bool,
                  refresh_days: int, max_comments: int) -> str:
    """Runs the extraction pipeline for one video. Returns 'ok', 'skip', or 'fail'."""
    tag      = f"[{idx}/{total}]"
    video_id = extract_video_id(url)
    if not video_id:
        print(f"{tag} [SKIP] Unrecognized URL: {url}")
        return "skip"

    video_dir = OUTPUT_DIR / video_id
    if not force and not needs_refresh(video_dir, refresh_days):
        print(f"{tag} [SKIP] {video_id} — fresh (<{refresh_days} days)")
        return "skip"

    # --- Early Filter: Shorts (URL patterns) ---
    if "/shorts/" in url.lower():
        print(f"{tag} [SKIP] (Short detected via URL): {url}")
        return "skip"

    print(f"{tag} [START] {video_id}")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Pre-fetch stats (requires YOUTUBE_API_KEY in .env)
    yt = {}
    try:
        yt = get_video_stats(video_id, YOUTUBE_API_KEY)
    except RuntimeError as e:
        if str(e) == "YOUTUBE_QUOTA_EXCEEDED":
            if idx == 1: # Only print this verbose warning once per run
                print(f"      [WARN] YouTube Data API Quota Exceeded. Some metadata (duration/views) will be missing.")
        else:
            raise
    
    # --- Early Filter: Disabled Comments ---
    if yt and yt.get("comments_disabled"):
        print(f"      [SKIP] Comments are disabled. Skipping.")
        return "skip"

    # --- Early Filter: Shorts (Duration < 30s) ---
    if yt and 0 < yt.get("duration", 0) < 30:
        print(f"      [SKIP] Video length is {yt['duration']}s. Skipping (under 30s).")
        return "skip"

    n_total = 0
    if yt:
        n_total = yt.get("comment_count", 0)
        cap     = max_comments if max_comments > 0 else n_total
        est     = f" | Est. ~{fmt_duration(cap / 2.5)}" if cap > 0 else ""
        print(f"  {yt['title'][:50]} | "
              f"{_short_num(n_total)} comments "
              f"{_short_num(yt['view_count'])} views"
              f"{est}")

    status: dict[str, str] = {}
    transcript: str | None = None
    comments: list[dict] | None = None
    any_error = False

    # Results tracking
    results = []

    # --- Transcript ---
    t_path = video_dir / "transcript.txt"
    if not force and t_path.exists():
        print("  [SKIP] Transcript already exists")
        status["transcript"] = "ok"
        results.append("Transcript (cached)")
        # Load transcript for meta-recording
        transcript = t_path.read_text(encoding="utf-8")
    else:
        p_fetch = (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable)
        transcript, err = with_retry(_fetch_transcript, video_id, label="Transcript", permanent_exceptions=p_fetch)
        if err:
            status["transcript"] = f"error: {err}"
            print(f"  [WARN] Transcript: {err}")
            any_error = True
        else:
            (video_dir / "transcript.txt").write_text(format_transcript(transcript, video_id, url), encoding="utf-8")
            status["transcript"] = "ok"
            results.append("Transcript")

    # --- Comments ---
    c_path = video_dir / "comments.json"
    needs_comment_fetch = True
    
    if not force and c_path.exists():
        try:
            c_data = json.loads(c_path.read_text(encoding="utf-8"))
            comments = c_data.get("comments", [])
            # If we already have enough comments (or all of them), we skip re-fetching
            if max_comments > 0 and len(comments) >= max_comments:
                print(f"  [SKIP] Comments already satisfy max_comments ({len(comments)})")
                status["comments"] = "ok"
                results.append(f"Comments({len(comments)}) (cached)")
                needs_comment_fetch = False
            elif max_comments == 0 and n_total > 0 and len(comments) >= (n_total * 0.95):
                # Allow a 5% margin for deleted/filtered comments if fetching "all"
                print(f"  [SKIP] Comments appear complete ({len(comments)}/{n_total})")
                status["comments"] = "ok"
                results.append(f"Comments({len(comments)}) (cached)")
                needs_comment_fetch = False
        except Exception:
            pass # Re-fetch if corrupted

    if needs_comment_fetch:
        comments, err = with_retry(_fetch_comments, video_id, max_comments, n_total, video_dir, url, label="Comments")
        if err:
            status["comments"] = f"error: {err}"
            print(f"  [WARN] Comments: {err}")
            any_error = True
        else:
            # Final save to ensure everything is locked in
            (video_dir / "comments.json").write_text(format_comments_json(comments, video_id, url), encoding="utf-8")
            status["comments"] = "ok"
            results.append(f"Comments({len(comments)})")

    
    _write_meta(video_dir, video_id, url, transcript, comments, status)
    
    if results:
        print(f"  [OK] {', '.join(results)}")
    
    print(f"{tag} [DONE] {video_id}\n")
    return "fail" if any_error else "ok"


# =============================================================================
# Entry point
# =============================================================================

def _print_help() -> None:
    w      = 62
    border = "+" + "-" * w + "+"
    def row(text=""):   # noqa: E306
        print(f"| {text:<{w - 2}} |")
    print(border)
    row("  YouTube Data Pipeline")
    row("  Reads video.txt and extracts transcripts and comments")
    row("  for each video.")
    print(border)
    row()
    row("  USAGE")
    row("    python pipeline.py [options]")
    row()
    row("  OPTIONS")
    row("    --force              Re-fetch even if data is fresh")
    row(f"    --refresh-days N     Staleness threshold in days (default: {REFRESH_AFTER_DAYS})")
    row("    --max-comments N     Cap comments per video (default: 0 = all)")
    row("    -h, --help           Show this help panel")
    row()
    row("  EXAMPLES")
    row("    python pipeline.py")
    row("    python pipeline.py --force")
    row("    python pipeline.py --refresh-days 7")
    row("    python pipeline.py --max-comments 500")
    row()
    row("  OUTPUT  output/<video_id>/")
    row("    transcript.txt    full video transcript")
    row("    comments.json     comments with full metadata")
    row("    meta.json         extraction stats + status")
    row()
    row("  SETUP")
    row("    1. Add YouTube URLs to video.txt (one per line)")
    row("    2. python pipeline.py")
    row()
    print(border)


def _parse_args() -> argparse.Namespace:
    if "-h" in sys.argv or "--help" in sys.argv:
        _print_help()
        sys.exit(0)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--refresh-days",  type=int, default=REFRESH_AFTER_DAYS, metavar="N")
    parser.add_argument("--max-comments",  type=parse_count, default=MAX_COMMENTS,        metavar="N", help="Max comments (e.g. 100, 5K, 1M)")
    parser.add_argument("--workers",       type=int, default=4,                  metavar="N")
    return parser.parse_args()


def _load_urls() -> list[str]:
    path = Path(VIDEO_LIST_FILE)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[ERR] {VIDEO_LIST_FILE} not found. Please create it and add YouTube URLs.")
        sys.exit(1)

    seen: set[str] = set()
    urls: list[str] = []
    
    for line in content.splitlines():
        url = line.strip()
        if url and not url.startswith("#") and url not in seen:
            seen.add(url)
            urls.append(url)
            
    if not urls:
        print(f"[ERR] No valid URLs found in {VIDEO_LIST_FILE}.")
        sys.exit(1)
        
    return urls


def main() -> None:
    args = _parse_args()
    urls = _load_urls()
    OUTPUT_DIR.mkdir(exist_ok=True)

    flags = []
    if args.force:                              flags.append("--force")
    if args.refresh_days != REFRESH_AFTER_DAYS: flags.append(f"--refresh-days {args.refresh_days}")
    if args.max_comments != MAX_COMMENTS:       flags.append(f"--max-comments {args.max_comments}")

    flag_str = f"  [{' '.join(flags)}]" if flags else ""
    cap_str  = f"{args.max_comments:,}" if args.max_comments > 0 else "all"
    print(f"[PIPELINE] {len(urls)} URL(s){flag_str} -> ./{OUTPUT_DIR}/"
          f"  [refresh: {args.refresh_days}d | comments: {cap_str}]\n")

    results = {"ok": 0, "skip": 0, "fail": 0}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    workers = args.workers if hasattr(args, 'workers') else 4

    print(f"[PIPELINE] Starting extraction with {workers} workers...\n")

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_url = {
                executor.submit(process_video, url, i, len(urls), args.force, args.refresh_days, args.max_comments): url 
                for i, url in enumerate(urls, 1)
            }
            for future in as_completed(future_to_url):
                outcome = future.result()
                results[outcome] += 1
    except KeyboardInterrupt:
        print("\n\n[HALT] Interrupt detected! Shutting down workers...")
        # Python 3.9+ cancel_futures=True will prevent new tasks from starting
        # and attempt to stop existing ones if possible.
        executor.shutdown(wait=False, cancel_futures=True)
        print("[HALT] Pipeline stopped. Progress saved to checkpoints.")
        sys.exit(1)

    print("=" * 50)
    print(f"[SUMMARY] Processed: {results['ok']}  |  "
          f"Skipped: {results['skip']}  |  Failed: {results['fail']}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[HALT] Pipeline interrupted by user.")
        sys.exit(0)
