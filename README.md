# YouTube Data Pipeline

A sequential pipeline that reads YouTube URLs from `video.txt`, fetches transcripts and comments per video, summarizes them using an LLM, and saves everything into a structured output folder.

## Project Structure

```
YTCBERT/
├── pipeline.py          # Main pipeline script
├── video.txt            # Input: one YouTube URL per line
├── requirements.txt     # Python dependencies
├── .env                 # API keys (gitignored)
├── .env.example         # Template for .env
├── .gitignore
└── output/
    └── <video_id>/
        ├── transcript.txt   # Full video transcript (with header)
        ├── comments.json    # Top comments with full metadata
        ├── summary.txt      # LLM-generated summary (with header)
        └── meta.json        # Extraction stats + status
```

## Setup

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
copy .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

## Usage

Add YouTube URLs to `video.txt` (one per line, `#` prefixes are treated as comments):

```
# My videos
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://youtu.be/XXXXXX
```

Then run the pipeline:

```powershell
python pipeline.py                # Normal run
python pipeline.py --force        # Re-fetch all, ignoring freshness
python pipeline.py --no-summary   # Skip LLM summarization
```

## Behavior

| Scenario | Outcome |
|---|---|
| Video already processed, data < 30 days old | Skipped |
| Video already processed, data > 30 days old | Re-fetched |
| Transcript disabled / unavailable | Logged as warning, continues to comments |
| Comments disabled | Logged as warning, continues to summary |
| No API key set | Summary skipped, rest saves normally |
| Network error | Retries up to 3× with exponential backoff |
| Duplicate URL in `video.txt` | Deduplicated automatically |

## comments.json Schema

```json
{
  "meta": {
    "video_id": "dQw4w9WgXcQ",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "extracted_at": "2026-03-09T06:22:45+00:00",
    "count": 500,
    "total_words": 2200,
    "avg_words_per_comment": 4.4,
    "max_words": 31,
    "min_words": 1
  },
  "comments": [
    {
      "index": 1,
      "text": "can confirm: he never gave us up",
      "author": "@YouTube",
      "votes": "193k",
      "time": "10 months ago",
      "is_reply": false,
      "is_favorited": true
    }
  ]
}
```

**Loading for BERT pipelines:**
```python
import json

data = json.load(open("comments.json"))

# Top-level comments only, sorted by engagement
top = sorted(
    [c for c in data["comments"] if not c["is_reply"]],
    key=lambda c: int(c["votes"].replace("k","000").replace(".","")) if c["votes"].isdigit() else 0,
)
texts = [c["text"] for c in top]
```

## Configuration

Edit the constants at the top of `pipeline.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_COMMENTS` | `500` | Max comments fetched per video |
| `REFRESH_AFTER_DAYS` | `30` | Days before re-fetching a video |
| `MAX_TRANSCRIPT_CHARS` | `12000` | Transcript chars sent to LLM (~3k tokens) |
| `MAX_COMMENTS_CHARS` | `8000` | Comment chars sent to LLM (~2k tokens) |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model used for summarization |
| `RETRY_ATTEMPTS` | `3` | Max retries on transient network errors |

## Dependencies

- [`youtube-transcript-api`](https://github.com/jdepoix/youtube-transcript-api) — transcript fetching
- [`youtube-comment-downloader`](https://github.com/egbertbouman/youtube-comment-downloader) — comment scraping (no API key needed)
- [`openai`](https://github.com/openai/openai-python) — LLM summarization
- [`python-dotenv`](https://github.com/theskumar/python-dotenv) — `.env` file loading