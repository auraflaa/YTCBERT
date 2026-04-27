"""
Model Training/prepare_dataset.py
--------------------------------
Aggregates finalized summaries and raw comments from the Data Pre Processing folder
and generates a JSONL dataset optimized for FLAN-T5 fine-tuning.
"""

import json
from pathlib import Path

# --- CONFIG ---
MT_DIR = Path(__file__).resolve().parent
ROOT_DIR = MT_DIR.parent
SUMMARY_DIR = ROOT_DIR / "Data_Preprocessing" / "summaries"
SOURCE_DIR = ROOT_DIR / "Data_Preprocessing" / "cleaned_output"
OUTPUT_FILE = MT_DIR / "summarization_dataset.jsonl"

def prepare():
    if not SUMMARY_DIR.exists():
        print(f"Error: Summaries directory not found at {SUMMARY_DIR}")
        return

    summary_files = list(SUMMARY_DIR.glob("*.json"))
    print(f"Aggregating {len(summary_files)} summaries...")

    data_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for s_path in summary_files:
            try:
                # 1. Load Summary (Teacher Output)
                with open(s_path, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                
                vid = s_data.get("video_id")
                summary_text = s_data.get("summary")
                
                if not vid or not summary_text:
                    continue

                # 2. Load Raw Comments (Student Input)
                c_path = SOURCE_DIR / vid / "cleaned_comments.json"
                if not c_path.exists():
                    continue
                
                with open(c_path, "r", encoding="utf-8") as f:
                    c_data = json.load(f)
                
                # Combine comment text (Limit to first 200 items to match teacher context)
                comments = c_data.get("comments", [])[:200]
                source_text = " ".join([c.get("text", "") for c in comments])
                
                # 3. Write T5 Pair
                entry = {
                    "input": f"Summarize audience comments: {source_text}",
                    "output": summary_text
                }
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                data_count += 1
                
            except Exception as e:
                print(f"Skipping {s_path.name}: {e}")

    print("\nDataset preparation complete!")
    print(f"Total pairs generated: {data_count}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare()
