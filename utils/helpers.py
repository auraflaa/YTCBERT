"""
utils/helpers.py
----------------
General-purpose helpers: URL parsing, cache freshness checks,
error sanitisation, duration formatting, and YouTube API stats.
"""

import json
import re
import urllib.request
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# URL / ID helpers
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str | None:
    """Return the 11-char YouTube video ID from any supported URL format."""
    match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------

def needs_refresh(video_dir: Path, refresh_days: int) -> bool:
    """True when the folder is missing, incomplete, stale, or corrupt."""
    required = ("transcript.txt", "comments.json", "meta.json")
    if not video_dir.exists() or any(not (video_dir / f).exists() for f in required):
        return True
    try:
        meta = json.loads((video_dir / "meta.json").read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(meta["extracted_at"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts) > timedelta(days=refresh_days)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Error sanitisation
# ---------------------------------------------------------------------------

def clean_err(exc: Exception | str) -> str:
    """
    Returns a concise, human-readable error string.
    Strips verbose JSON bodies from OpenAI errors and truncates long messages.
    """
    msg = str(exc)
    inner = re.search(r"'message':\s*'([^']+)'", msg)
    if inner:
        msg = inner.group(1)
    return msg[:117] + "..." if len(msg) > 120 else msg


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    """Format a number of seconds as a human-readable string, e.g. '1h 4m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


# ---------------------------------------------------------------------------
# YouTube Data API v3
# ---------------------------------------------------------------------------

def get_video_stats(video_id: str, api_key: str) -> dict:
    """
    Fetches title, comment_count, view_count, like_count for a video.
    Returns an empty dict if the key is missing or the request fails.
    """
    if not api_key:
        return {}
    try:
        url = (
            f"https://www.googleapis.com/youtube/v3/videos"
            f"?part=statistics,snippet&id={video_id}&key={api_key}"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        items = data.get("items", [])
        if not items:
            return {}
        snippet = items[0].get("snippet", {})
        stats   = items[0].get("statistics", {})
        return {
            "title":         snippet.get("title", ""),
            "comment_count": int(stats.get("commentCount", 0)),
            "view_count":    int(stats.get("viewCount",    0)),
            "like_count":    int(stats.get("likeCount",    0)),
        }
    except Exception:
        return {}
# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def with_retry(fn, *args, attempts: int = 3, backoff: int = 2, label: str = "", permanent_exceptions: tuple = ()):
    """
    Calls fn(*args). Retries up to `attempts` times with exponential backoff.
    Handles RateLimit (429) errors with specific backoff.
    Returns (result, None) on success or (None, error_str) on failure.
    """
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args), None
        except permanent_exceptions as e:
            return None, clean_err(e)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RateLimit" in err_str or "quota" in err_str.lower()
            
            if attempt == attempts:
                return None, f"{label} failed after {attempts} attempts: {clean_err(e)}"
            
            # Intelligent wait for rate limits vs transient errors
            wait = (backoff ** attempt)
            if is_rate_limit:
                wait += 10 # Extra buffer for rate limits
                print(f"  [RATE LIMIT] {label} hit quota limit. Waiting {wait}s before retry {attempt+1}/{attempts}...")
            else:
                print(f"  [RETRY] {label} attempt {attempt}/{attempts} failed. Retrying in {wait}s...")
            
            time.sleep(wait)
    return None, f"{label} failed: {last_error}"


# ---------------------------------------------------------------------------
# Text / Content helpers
# ---------------------------------------------------------------------------

def strip_banner(text: str) -> str:
    """Strips the ASCII banner from the start of a transcript/summary if it exists."""
    if "======" in text:
        parts = text.split("========================================================================\n\n", 1)
        return parts[1] if len(parts) > 1 else text
    return text


def load_prompts(prompt_file: str = "prompt.txt"):
    """Loads system and user prompts from a file."""
    try:
        raw = Path(prompt_file).read_text(encoding="utf-8")
        parts = raw.split("## USER", 1)
        system = parts[0].replace("## SYSTEM", "").strip()
        user = parts[1].strip() if len(parts) > 1 else None
        return system, user
    except FileNotFoundError:
        return None, None
