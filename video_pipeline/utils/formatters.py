"""
utils/formatters.py
-------------------
Output file formatters: transcript header, comments JSON, summary header.
"""

import json
from datetime import datetime, timezone

from utils.stats import word_stats, comment_texts, parse_votes


# ---------------------------------------------------------------------------
# Shared banner
# ---------------------------------------------------------------------------

def _banner(title: str, video_id: str, url: str,
            extra_lines: list[str] | None = None) -> str:
    """Creates a consistent ASCII header block for output files."""
    sep  = "=" * 72
    now  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = [
        sep,
        f"  {title}",
        sep,
        f"  Video ID   : {video_id}",
        f"  URL        : {url}",
        f"  Extracted  : {now}",
    ]
    if extra_lines:
        body.extend(f"  {l}" for l in extra_lines)
    body.append(sep)
    return "\n".join(body) + "\n\n"


# ---------------------------------------------------------------------------
# File formatters
# ---------------------------------------------------------------------------

def format_transcript(text: str, video_id: str, url: str) -> str:
    """Prepend a self-describing header to the raw transcript text."""
    wc = len(text.split())
    lc = text.count("\n") + 1
    return _banner(
        "TRANSCRIPT", video_id, url,
        [f"Words      : {wc:,}", f"Lines      : {lc:,}"],
    ) + text


def format_comments_json(comments: list[dict], video_id: str, url: str) -> str:
    """Serialize comments to structured JSON with an embedded metadata header."""
    texts = comment_texts(comments)
    stats = word_stats(texts)
    payload = {
        "meta": {
            "video_id":              video_id,
            "url":                   url,
            "extracted_at":          datetime.now(timezone.utc).isoformat(),
            "count":                 stats["count"],
            "total_words":           stats["total_words"],
            "avg_words_per_comment": stats["avg_words"],
            "max_words":             stats["max_words"],
            "min_words":             stats["min_words"],
        },
        "comments": [
            {
                "index":        i,
                "text":         c.get("text", ""),
                "author":       c.get("author", ""),
                "votes":        c.get("votes", "0"),
                "time":         c.get("time", ""),
                "is_reply":     c.get("reply", False),
                "is_favorited": c.get("heart", False),
            }
            for i, c in enumerate(comments, 1)
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_summary(summary: str, video_id: str, url: str,
                   llm_model: str, transcript_words: int,
                   comment_count: int) -> str:
    """Prepend a self-describing header to the LLM summary text."""
    return _banner(
        "LLM SUMMARY", video_id, url,
        [
            f"Model      : {llm_model}",
            f"Based on   : {transcript_words:,} transcript words, {comment_count:,} comments",
        ],
    ) + summary + "\n"
