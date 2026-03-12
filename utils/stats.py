"""
utils/stats.py
--------------
Statistics and metadata builders for transcripts and comments.
"""

import re
from collections import Counter
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

def word_stats(texts: list[str]) -> dict:
    """Returns count, total/avg/max/min word counts for a list of strings."""
    counts = [len(t.split()) for t in texts]
    return {
        "count":       len(counts),
        "total_words": sum(counts),
        "avg_words":   round(sum(counts) / len(counts), 2) if counts else 0.0,
        "max_words":   max(counts, default=0),
        "min_words":   min(counts, default=0),
    }


def comment_texts(comments: list[dict]) -> list[str]:
    """Extract the plain text field from a list of comment dicts."""
    return [c["text"] for c in comments if c.get("text")]


# ---------------------------------------------------------------------------
# Vote parsing
# ---------------------------------------------------------------------------

def parse_votes(votes_str: str) -> int:
    """Convert YouTube vote strings ('193k', '1.4k', '547') to integers."""
    v = str(votes_str).strip().lower().replace(",", "")
    try:
        if v.endswith("k"):
            return int(float(v[:-1]) * 1_000)
        if v.endswith("m"):
            return int(float(v[:-1]) * 1_000_000)
        return int(float(v))
    except (ValueError, AttributeError):
        return 0


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def transcript_meta(transcript: str | None) -> dict:
    """Build rich metadata dict for a transcript string."""
    if not transcript:
        return {"available": False}
    
    words = transcript.split()
    lines = [l for l in transcript.splitlines() if l.strip()]
    duration_mins = round(len(words) / 130, 1)  # ~130 wpm spoken

    # Simple keyword extraction (remove punctuation, lowercased, >2 chars, not stopword)
    STOPWORDS = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", 
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", 
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", 
        "an", "will", "my", "one", "all", "would", "there", "their", "what", 
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", 
        "take", "people", "into", "year", "your", "good", "some", "could", 
        "them", "see", "other", "than", "then", "now", "look", "only", "come", 
        "its", "over", "think", "also", "back", "after", "use", "two", "how",
        "our", "work", "first", "well", "way", "even", "new", "want", "because",
        "any", "these", "give", "day", "most", "us", "are", "is", "was", "were",
        "been", "being", "has", "had", "having", "does", "did", "doing",
        "am", "im", "dont", "cant", "that", "thats", "it", "its", "too", "very",
        "really", "going", "got", "much", "more", "those", "why", "where"
    }
    
    clean_words = re.findall(r'\b[a-z]{3,}\b', transcript.lower())
    meaningful  = [w for w in clean_words if w not in STOPWORDS]
    top_10      = [word for word, count in Counter(meaningful).most_common(10)]

    return {
        "available":                True,
        "word_count":               len(words),
        "unique_word_count":        len(set(clean_words)),
        "vocabulary_richness":      round(len(set(clean_words)) / len(words), 3) if words else 0.0,
        "line_count":               len(lines),
        "char_count":               len(transcript),
        "avg_words_per_line":       round(len(words) / len(lines), 1) if lines else 0.0,
        "estimated_duration_mins":  duration_mins,
        "top_keywords":             top_10,
    }


def comments_meta(comments: list[dict] | None) -> dict:
    """Build rich metadata dict for a list of comment dicts."""
    if not comments:
        return {"available": False, "count": 0}

    texts       = comment_texts(comments)
    w_counts    = [len(t.split()) for t in texts]
    vote_values = [parse_votes(c.get("votes", 0)) for c in comments]
    top_5 = [
        {
            "vote_rank": rank,
            "author":    c.get("author", ""),
            "votes":     c.get("votes", "0"),
            "text":      c.get("text", "")[:120],
        }
        for rank, c in enumerate(
            sorted(comments, key=lambda x: parse_votes(x.get("votes", 0)), reverse=True)[:5],
            start=1,
        )
    ]

    return {
        "available":            True,
        "count":                len(comments),
        "top_level_count":      sum(1 for c in comments if not c.get("reply", False)),
        "reply_count":          sum(1 for c in comments if c.get("reply", False)),
        "unique_authors":       len({c.get("author", "") for c in comments}),
        "favorited_count":      sum(1 for c in comments if c.get("heart", False)),
        "total_votes":          sum(vote_values),
        "avg_votes_per_comment": round(sum(vote_values) / len(vote_values), 1) if vote_values else 0.0,
        "total_words":          sum(w_counts),
        "avg_words":            round(sum(w_counts) / len(w_counts), 2) if w_counts else 0.0,
        "max_words":            max(w_counts, default=0),
        "min_words":            min(w_counts, default=0),
        "top_5_by_votes":       top_5,
    }
