"""
Data Cleaning/clean_pipeline.py
------------------------------
Batch processes the YouTube dataset applying the 8-step cleaning logic.
"""

import os
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TaskProgressColumn
from rich.panel import Panel
from rich import box

# --- ROOT AWARENESS & CONFIG ---
CURR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURR_DIR.parent
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

# Standardized Paths
SOURCE_DIR = ROOT_DIR / "Video Pipeline" / "output"
CLEANED_DIR = CURR_DIR / "cleaned_output"

# Local Import
from processor import CommentProcessor

console = Console()

def clean_batch(source_dir: Path, target_dir: Path, force: bool = False):
    if not source_dir.exists():
        console.print(f"[red]Error: {source_dir} not found.[/red]")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    video_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    processor = CommentProcessor()
    
    stats = {
        "total_videos": len(video_dirs),
        "total_raw_comments": 0,
        "total_cleaned_comments": 0,
        "skipped_videos": 0
    }

    console.print(f"\n[bold blue]Starting Cleaning Pipeline...[/bold blue]\n")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing videos...", total=len(video_dirs))
        
        try:
            for v_dir in video_dirs:
                # 0. Skip if already processed and not forced
                v_out_dir = target_dir / v_dir.name
                if not force and (v_out_dir / "cleaned_comments.json").exists():
                    progress.advance(task)
                    continue

                c_path = v_dir / "comments.json"
                if not c_path.exists():
                    stats["skipped_videos"] += 1
                    progress.advance(task)
                    continue

                try:
                    # 1. Load raw data
                    with open(c_path, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                    
                    raw_comments = raw_data.get("comments", [])
                    stats["total_raw_comments"] += len(raw_comments)

                    # 2. Process
                    cleaned_comments = processor.process_comments(raw_comments)
                    stats["total_cleaned_comments"] += len(cleaned_comments)

                    # 3. Save to dedicated output folder
                    v_out_dir.mkdir(exist_ok=True)
                    
                    save_data = {
                        "video_id": raw_data.get("video_id"),
                        "url": raw_data.get("url"),
                        "cleaning_stats": {
                            "raw_count": len(raw_comments),
                            "cleaned_count": len(cleaned_comments),
                            "removed": len(raw_comments) - len(cleaned_comments)
                        },
                        "comments": cleaned_comments
                    }
                    
                    with open(v_out_dir / "cleaned_comments.json", "w", encoding="utf-8") as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    console.print(f"  [red]Error processing {v_dir.name}: {e}[/red]")
                
                progress.advance(task)
        except KeyboardInterrupt:
            console.print("\n[bold red][HALT] Cleaning interrupted. Showing partial summary...[/bold red]")

    # --- Summary Report ---
    report = Table(title="Cleaning Pipeline Summary", box=box.ROUNDED)
    report.add_column("Metric", style="cyan")
    report.add_column("Value", justify="right", style="magenta")
    
    report.add_row("Total Folders Scanned", str(stats["total_videos"]))
    report.add_row("Raw Comments Analyzed", f"{stats['total_raw_comments']:,}")
    report.add_row("Quality Pool Size", f"{stats['total_cleaned_comments']:,}")
    
    reduction = 0
    if stats["total_raw_comments"] > 0:
        reduction = (1 - (stats["total_cleaned_comments"] / stats["total_raw_comments"])) * 100
    
    report.add_row("Noise Reduction %", f"{reduction:.1f}%")
    
    console.print("\n", report, "\n")
    console.print(f"[dim]Cleaned files saved in: {target_dir}[/dim]\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Data Cleaning Pipeline")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cleaned files")
    args = parser.parse_args()

    clean_batch(SOURCE_DIR, CLEANED_DIR, force=args.force)
