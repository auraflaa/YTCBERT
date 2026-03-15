# YouTube Data Pipeline (YTCBERT)

A high-performance, two-step pipeline designed to extract YouTube transcripts and comments at scale, followed by AI-powered summarization and model comparison.

## 🚀 Optimized Workflow

1.  **Extract**: Use `pipeline.py` to fetch raw data (transcripts + comments).
2.  **Summarize**: Use `summarize_data.py` to process extracted data with LLMs.

---

## 📂 Project Structure

```text
YTCBERT/
├── pipeline.py          # Step 1: High-speed data extraction
├── summarize_data.py    # Step 2: Multithreaded batch summarization
├── compare_models.py    # Side-by-side LLM performance comparison
├── utils/               # Core logic (LLM, Formatters, Helpers)
├── video.txt            # Input: List of YouTube URLs
├── prompt.txt           # Input: System/User prompt templates
├── models.txt           # Input: Model definitions for comparison
├── output/              # Data Store: Per-video folders
│   └── <video_id>/
│       ├── transcript.txt   # Cleaned transcript
│       ├── comments.json    # Metadata-rich comment storage
│       ├── summary.txt      # AI Summary
│       └── meta.json        # Extraction stats & status
└── comparisons/         # Generated comparison reports
```

---

## ⚙️ Features & Optimizations

*   **Multithreaded Processing**: Process dozens of summaries or model comparisons in parallel using `--workers`.
*   **Granular Resumption**: If interrupted, the pipeline skips individual files (`transcript.txt`, `comments.json`) already successfully downloaded.
*   **Video Filtering**: Automatically skips **YouTube Shorts** (via URL or duration < 60s) and videos with **disabled comments** to ensure data quality for training.
*   **Smart Fallbacks**: Automatically switches between `GOOGLE_API_KEY` (Gemini) and `LLM_API_KEY` (OpenAI/GPT) depending on availability.
*   **Engagement-First**: Comments are fetched in order of popularity, perfect for BERT-based sentiment analysis.

---

## 🛠️ Setup

1.  **Environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Copy `.env.example` to `.env` and add your `GOOGLE_API_KEY` or `OPENROUTER_API_KEY`.

---

## 📖 Usage Guide

### 1. Data Extraction
Add URLs to `video.txt` and run:
```bash
python pipeline.py --max-comments 5000
```
*   **Resume Capability**: Run again any time; it will only fetch missing or stale data.
*   **Flags**: Use `--force` to ignore the 30-day cache.

### 2. AI Summarization
Generate summaries for your extracted data:
```bash
python summarize_data.py --workers 10
```
*   **Performance**: Use `--workers` to set concurrency (default: 5).
*   **Flexibility**: Specify models with `--model gemini-1.5-flash`.

### 3. Model Comparison
Evaluate different LLM providers side-by-side:
```bash
python compare_models.py --video <ID> --workers 3
```
*   Results are saved as a markdown report in `comparisons/report_<ID>.md`.

---

## 📊 Data Schema (`comments.json`)

The extracted data is structured for immediate use in NLP pipelines:

```json
{
  "meta": { "video_id": "...", "url": "...", "count": 10000 },
  "comments": [
    {
      "text": "Insightful breakdown!",
      "author": "@user",
      "votes": "1.2k",
      "time": "2 days ago"
    }
  ]
}
```

---

## 🔧 Configuration Constants

| Constant | File | Default | Description |
|---|---|---|---|
| `MAX_COMMENTS` | `pipeline.py` | `10000` | Comment cap per video |
| `REFRESH_AFTER_DAYS` | `pipeline.py` | `30` | Cache expiry |
| `MAX_TRANSCRIPT_CHARS` | `summarize_data.py` | `12000` | LLM context limit |
| `MAX_COMMENTS_CHARS` | `summarize_data.py` | `8000` | LLM context limit |
| `DEFAULT_WORKERS` | `summarize_data.py` | `5` | Concurrency level |