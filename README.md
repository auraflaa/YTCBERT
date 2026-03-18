# YouTube Data Pipeline (YTCBERT)

A high-performance system designed to extract YouTube transcripts and comments at scale, followed by hierarchical AI-powered summarization and dataset exportation for fine-tuning models like FLAP-T5.

## 🚀 Optimized Workflow

1.  **Discover**: Use `video_pipeline/discover_videos.py` to find diverse English content.
2.  **Extract**: Use `video_pipeline/pipeline.py` to fetch raw transcripts and engagement-rich comments.
3.  **Summarize**: Use `video_pipeline/summarize_data.py` to process long videos using hierarchical condensation.
4.  **Visualize**: Use `video_pipeline/visualize_diversity.py` to analyze your dataset balance.
5.  **Export**: Use `video_pipeline/export_dataset.py` to generate training-ready JSONL datasets.

---

## 📂 Project Structure

```text
YTCBERT/
├── video_pipeline/       # Descriptive home for all processing code (The Engine)
│   ├── discover_videos.py    # Step 0: Goal-aware English video discovery
│   ├── pipeline.py           # Step 1: High-speed data extraction (Checkpointed)
│   ├── summarize_data.py     # Step 2: Hierarchical Map-Reduce summarization
│   ├── visualize_diversity.py# Step 3: Dataset diversity & distribution dashboard
│   ├── export_dataset.py     # Step 4: Export curation to JSONL (FLAP-T5 format)
│   ├── verify_videos.py      # Utility: Validate status & categories of URLs
│   ├── clean_video_links.py  # Utility: Category-aware link cleaning & pruning
│   ├── video.txt             # Input: Organized list of YouTube URLs (with # Category:)
│   └── utils/                # Core logic (LLM, Helpers, Formatters)
├── output/               # Data Store: Per-video structured folders
├── backups/              # Automatic backups generated during maintenance
├── .env                  # Environment variables (API Keys)
└── requirements.txt      # Project dependencies
```

---

## ⚙️ Features & Optimizations

*   **Goal-Aware Discovery**: Treats `--count` as a total target; automatically balances niches based on existing links.
*   **Crash-Proof Resilience**: Graceful `Ctrl+C` handling across all scripts; flushes progress to disk before exiting.
*   **Hierarchical Summarization**: Map-Reduce architecture for long videos, ensuring full context coverage.
*   **Checkpointing & Resumption**: Smart resumption for both discovery and extraction; picks up exactly where it left off.
*   **Category-Aware Maintenance**: Maintenance scripts preserve and organize your list by hobbyist-friendly niche headers.

---

## 🛠️ Setup

1.  **Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Or .\venv\Scripts\Activate.ps1 on Windows
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Create a `.env` file in the project root with:
    ```env
    LLM_API_KEY=your_openai_or_gemini_key
    YOUTUBE_API_KEY=your_youtube_data_api_key (Required for discovery/visuals)
    ```

---

## 📖 Usage Guide

> [!IMPORTANT]
> All scripts should be run from the **project root** directory.

### 1. Broad Discovery (Goal-Aware & Quality-Filtered)
Find new English content. The script intelligently balances niches and strictly filters out "junk" videos (Shorts or those with no comments) before they ever reach your dataset.

Options:
- `--count N`: The total target number of videos for your list.
- `--min-comments N`: Drop videos with fewer than `N` comments (default: 10).
- `--min-length M`: Drop videos shorter than `M` minutes (default: 1.0).

```bash
python video_pipeline/discover_videos.py --count 1000 --min-comments 20 --min-length 1.5
```
> [!TIP]
> **Customizing Categories**: You can easily edit or add your own niches by modifying the `CATEGORIES` dictionary at the top of `video_pipeline/discover_videos.py`.

### 2. Data Extraction (Resumable)
Gather transcripts and comments. Safe to interrupt and resume:
```bash
python video_pipeline/pipeline.py --workers 4 --max-comments 5000
```

### 3. Hierarchical Summarization
Generate high-density summaries for long-form content:
```bash
python video_pipeline/summarize_data.py --workers 5 --workers-inner 3
```

### 4. Diversity Visualization
Analyze your dataset balance and engagement distribution:
```bash
python video_pipeline/visualize_diversity.py
```
> [!TIP]
> **Interactive Reports:** You can generate a beautiful, interactive Plotly dashboard. Use `--show-report` to view it instantly in your browser (temporary), or `--export-report` to save it to the `YTCBERT/reports/` folder!

### 5. Dataset Export
Build your training-ready JSONL file:
```bash
python video_pipeline/export_dataset.py --out my_t5_dataset.jsonl
```

---

## 🔧 Maintenance Utilities

*   **Verify**: `python video_pipeline/verify_videos.py` (High-concurrency API Batch Engine. Automatically flags PRIVATE, DELETED, RESTRICTED, Shorts (<30s), and Zero-Comment videos!).
*   **Clean**: `python video_pipeline/clean_video_links.py --apply-report` (Instantly reads the verify report to natively purge all invalid/broken/short URLs).
*   **Audit**: `python video_pipeline/prune_vids.py` (Scans `output/` folders post-extraction securely backs up and removes empty/dud data entries).