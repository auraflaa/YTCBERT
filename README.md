# YouTube Data Pipeline (YTCBERT)

A high-performance system designed to extract YouTube transcripts and comments at scale, followed by hierarchical AI-powered summarization and dataset exportation for fine-tuning models like FLAP-T5.

## 🚀 Optimized Workflow

1.  **Discover**: Use `video_pipeline/discover_videos.py` to find diverse English content.
2.  **Extract**: Use `video_pipeline/pipeline.py` to fetch raw transcripts and engagement-rich comments.
3.  **Summarize**: Use `video_pipeline/summarize_data.py` to process long videos using hierarchical condensation.
4.  **Export**: Use `video_pipeline/export_dataset.py` to generate training-ready JSONL datasets.

---

## 📂 Project Structure

```text
YTCBERT/
├── video_pipeline/      # Descriptive home for all processing code (The Engine)
│   ├── discover_videos.py   # Step 0: Automated English video discovery
│   ├── pipeline.py          # Step 1: High-speed data extraction (Checkpointed)
│   ├── summarize_data.py    # Step 2: Hierarchical Map-Reduce summarization
│   ├── export_dataset.py    # Step 3: Export curation to JSONL (FLAP-T5 format)
│   ├── verify_videos.py     # Utility: Validate status of URLs in video.txt
│   ├── remove_duplicates.py # Utility: Clean duplicates from video.txt
│   └── utils/               # Core logic (LLM, Helpers, Formatters)
├── video.txt            # Input: List of YouTube URLs
├── output/              # Data Store: Per-video structured folders
├── backups/             # Automatic backups of video.txt
└── .env                 # Environment variables (API Keys)
```

---

## ⚙️ Features & Optimizations

*   **Video Pipeline Engine**: Centralized, descriptive folder structure for easy maintenance and scaling.
*   **Pro Video Discovery**: Scalable English video mining with category-based rotation.
*   **Hierarchical Summarization**: Map-Reduce architecture for long videos, ensuring full context coverage.
*   **Checkpointing & Atomic Writes**: Resumable comment downloads and crash-safe JSON/Text saving.
*   **Multithreaded Processing**: Process dozens of videos or sub-batches in parallel.

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
    YOUTUBE_API_KEY=optional_key_for_rich_stats
    ```

---

## 📖 Usage Guide

> [!IMPORTANT]
> All scripts should be run from the **project root** directory.

### 1. Broad Discovery
Find new English content across high-quality categories:
```bash
python video_pipeline/discover_videos.py --count 25 --append
```

### 2. Data Extraction
Gather transcripts and comments for everything in `video.txt`:
```bash
python video_pipeline/pipeline.py --workers 4 --max-comments 5000
```

### 3. Hierarchical Summarization
Generate high-density summaries for long-form content:
```bash
python video_pipeline/summarize_data.py --workers 5 --workers-inner 3
```

### 4. Dataset Export
Build your training file:
```bash
python video_pipeline/export_dataset.py --out my_t5_dataset.jsonl
```

---

## 🔧 Maintenance Utilities

*   **Verify**: `python video_pipeline/verify_videos.py` (Checks availability).
*   **Deduplicate**: `python video_pipeline/remove_duplicates.py` (Cleans your list).