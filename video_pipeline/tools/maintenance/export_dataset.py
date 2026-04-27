"""
tools/maintenance/export_dataset.py
----------------------------------
Export Utility for FLAP-T5 Dataset Generation.
Aggregates raw data and hierarchical summaries from output/ into a JSONL file.
"""

import json
import argparse
import sys
import os
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.progress import track

# Local Imports
try:
    from utils.helpers import strip_banner
except ImportError:
    def strip_banner(x): return x

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Export hierarchical summarization results to JSONL.")
    parser.add_argument("--out", default="flap_t5_dataset.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    output_dir = ROOT_DIR / "output"
    if not output_dir.exists():
        console.print("[red][ERR] No output directory found.[/red]")
        return

    v_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if not v_dirs:
        console.print("[yellow]No data to export.[/yellow]")
        return

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT_DIR / out_path

    console.print(f"[bold blue]Exporting dataset for {len(v_dirs)} videos to {out_path}...[/bold blue]")

    exported_count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for v_dir in track(v_dirs, description="Processing videos..."):
            t_path = v_dir / "transcript.txt"
            c_path = v_dir / "comments.json"
            s_path = v_dir / "summary.txt"
            
            if not s_path.exists():
                continue

            try:
                transcript_text = t_path.read_text(encoding="utf-8") if t_path.exists() else ""
                
                comments_list = []
                if c_path.exists():
                    c_data = json.loads(c_path.read_text(encoding="utf-8"))
                    comments_list = c_data.get("comments", [])
                
                comments_text = "\n".join([f"- {c.get('text', '')}" for c in comments_list])
                source = f"TRANSCRIPT:\n{transcript_text}\n\nCOMMENTS:\n{comments_text}"
                
                target = strip_banner(s_path.read_text(encoding="utf-8"))

                example = {
                    "video_id": v_dir.name,
                    "source": source.strip(),
                    "target": target.strip()
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                exported_count += 1
            except Exception as e:
                console.print(f"[yellow][WARN] Failed to export {v_dir.name}: {e}[/yellow]")

    console.print(f"[bold green]Successfully exported {exported_count} examples.[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
