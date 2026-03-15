"""
YouTube Data Pipeline
---------------------
Reads YouTube URLs from video.txt and for each video:
  1. Skips if data is fresh (< REFRESH_AFTER_DAYS old) — override with --force
  2. Fetches the video transcript  (retries on transient errors)
  3. Fetches all comments          (retries on transient errors)
  4. Saves transcript.txt, comments.json, meta.json
     into output/<video_id>/
  5. Prints a run summary (processed / skipped / failed)

Usage:
  python pipeline.py [--force] [--refresh-days N] [--max-comments N]

Setup:
  pip install -r requirements.txt
  Copy .env.example -> .env and fill in LLM_API_KEY (and optionally YOUTUBE_API_KEY)
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
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
import langdetect
from langdetect.lang_detect_exception import LangDetectException

from utils.formatters import format_comments_json, format_transcript
from utils.helpers import clean_err, extract_video_id, fmt_duration, get_video_stats, needs_refresh, with_retry
from utils.stats import comment_texts, comments_meta, transcript_meta

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
VIDEO_LIST_FILE      = "video.txt"
OUTPUT_DIR           = Path("output")
MAX_COMMENTS         = 0        # 0 = fetch all; set > 0 to cap
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
                    total_hint: int = 0) -> list[dict]:
    """Fetches comments with a live graphical rich progress bar."""
    url        = f"https://www.youtube.com/watch?v={video_id}"
    downloader = YoutubeCommentDownloader()
    comments: list[dict] = []
    
    cap = max_comments if max_comments > 0 else total_hint

    with Progress(
        TextColumn("  [cyan]Fetching..."),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("•"),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[green]{task.fields[rate]} c/s"),
        TextColumn("•"),
        TextColumn("ETA"),
        TimeRemainingColumn(),
        transient=True  # Bar disappears cleanly when finished
    ) as progress:
        # Create a task. If cap is 0, bounded progress is disabled.
        task = progress.add_task("download", total=cap if cap > 0 else None, rate="0.0")
        start_time = time.time()
        
        for item in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
            text = item.get("text", "").strip()
            if not text:
                continue

            # Skip non-English comments
            try:
                if langdetect.detect(text) != 'en':
                    continue
            except LangDetectException:
                # Triggers on pure emojis, punctuation, gibberish etc.
                continue

            comments.append(item)
            elapsed = time.time() - start_time
            n = len(comments)
            rate_val = n / elapsed if elapsed > 0 else 0.0
            progress.update(task, advance=1, rate=f"{rate_val:.1f}")

            if max_comments > 0 and len(comments) >= max_comments:
                break

    return comments




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

    print(f"{tag} [START] {video_id}")
    video_dir.mkdir(parents=True, exist_ok=True)

    # Pre-fetch stats (requires YOUTUBE_API_KEY in .env)
    yt      = get_video_stats(video_id, YOUTUBE_API_KEY)
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
    comments, err = with_retry(_fetch_comments, video_id, max_comments, n_total, label="Comments")
    if err:
        status["comments"] = f"error: {err}"
        print(f"  [WARN] Comments: {err}")
        any_error = True
    else:
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
    row("  Reads video.txt and extracts transcript, comments,")
    row("  and LLM summary for each video.")
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
    row("    python pipeline.py --force --no-summary")
    row("    python pipeline.py --refresh-days 7")
    row("    python pipeline.py --max-comments 500")
    row()
    row("  OUTPUT  output/<video_id>/")
    row("    transcript.txt    full video transcript")
    row("    comments.json     comments with full metadata")
    row("    summary.txt       LLM-generated summary")
    row("    meta.json         extraction stats + status")
    row()
    row("  SETUP")
    row("    1. Add YouTube URLs to video.txt (one per line)")
    row("    2. Set LLM_API_KEY in .env")
    row("    3. python pipeline.py")
    row()
    print(border)


def _parse_args() -> argparse.Namespace:
    if "-h" in sys.argv or "--help" in sys.argv:
        _print_help()
        sys.exit(0)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--refresh-days",  type=int, default=REFRESH_AFTER_DAYS, metavar="N")
    parser.add_argument("--max-comments",  type=int, default=MAX_COMMENTS,        metavar="N")
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
    for i, url in enumerate(urls, 1):
        outcome = process_video(url, i, len(urls),
                                args.force,
                                args.refresh_days, args.max_comments)
        results[outcome] += 1

    print("=" * 50)
    print(f"[SUMMARY] Processed: {results['ok']}  |  "
          f"Skipped: {results['skip']}  |  Failed: {results['fail']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
