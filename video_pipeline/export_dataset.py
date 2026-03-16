"""
Export Utility for FLAP-T5 Dataset Generation (Step 4).

Aggregates raw data and hierarchical summaries from output/ into a JSONL file.
Formatted as (source, target) pairs ideal for T5-based fine-tuning.
"""

import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import track

# Configuration Constants
OUTPUT_DIR = Path("output")
DEFAULT_OUTFILE = "flap_t5_dataset.jsonl"

console = Console()

def main():
    """
    Main execution loop for dataset exportation.
    Iterates through video directories, parses raw data, and writes to JSONL.
    """
    parser = argparse.ArgumentParser(
        description="Export hierarchical summarization results to JSONL for FLAP-T5 training."
    )
    parser.add_argument("--out", default=DEFAULT_OUTFILE, help=f"Output JSONL file (default: {DEFAULT_OUTFILE})")
    args = parser.parse_args()

    # Find all video directories (named by video ID) in the output folder
    v_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    if not v_dirs:
        console.print("[red][ERR] No data found in output/ directory. Run pipeline.py first.[/red]")
        return

    console.print(f"[bold blue]Exporting dataset from {len(v_dirs)} videos to {args.out}...[/bold blue]")

    exported_count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        # iterate through each video's gathered data
        for v_dir in track(v_dirs, description="Processing videos..."):
            t_path = v_dir / "transcript.txt"      # Raw transcript lines
            c_path = v_dir / "comments.json"      # Raw JSON comments
            s_path = v_dir / "summary.txt"       # Final hierarchical summary (Target)
            
            # Skip videos that don't have a final summary yet
            if not s_path.exists():
                continue

            try:
                # 1. Prepare Source: Full raw transcript
                transcript_text = t_path.read_text(encoding="utf-8") if t_path.exists() else ""
                
                # 2. Prepare Source: All collected comments formatted as a list
                comments_data = {}
                if c_path.exists():
                    # Load the atomic JSON checkpoint
                    comments_data = json.loads(c_path.read_text(encoding="utf-8"))
                
                comments_list = comments_data.get("comments", [])
                comments_text = "\n".join([f"- {c.get('text', '')}" for c in comments_list])

                # Combine into a single source string for the LLM input
                source = f"TRANSCRIPT:\n{transcript_text}\n\nCOMMENTS:\n{comments_text}"
                
                # 3. Prepare Target: The ground-truth summary
                target = s_path.read_text(encoding="utf-8")

                # Clean target (remove debug banners/metadata if present)
                if "========================================================================\n\n" in target:
                     target = target.split("========================================================================\n\n", 1)[1]

                # Construct the JSONL example
                example = {
                    "video_id": v_dir.name,
                    "source": source.strip(),
                    "target": target.strip()
                }

                # Write as a single line in the JSONL file
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                exported_count += 1

            except Exception as e:
                console.print(f"[yellow][WARN] Failed to export {v_dir.name}: {e}[/yellow]")

    console.print(f"[bold green]Successfully exported {exported_count} examples to {args.out}[/bold green]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red][HALT] Export interrupted by user.[/bold red]")
        sys.exit(0)
