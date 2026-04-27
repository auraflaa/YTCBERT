"""
tools/maintenance/dataset_stats.py
----------------------------------
Provides a comprehensive overview of the extracted YouTube dataset.
"""

import json
import sys
import os
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.progress import track

def main():
    console = Console()
    output_dir = ROOT_DIR / "output"
    
    if not output_dir.exists():
        console.print("[red]No output directory found. Have you run the pipeline yet?[/red]")
        return
        
    v_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    if not v_dirs:
        console.print("[yellow]Output directory is empty.[/yellow]")
        return
        
    console.print(f"[bold blue]Scanning data for {len(v_dirs)} videos...[/bold blue]\n")
    
    stats = {
        "videos": len(v_dirs),
        "transcripts": 0,
        "transcript_lines": 0,
        "comments_files": 0,
        "total_comments": 0,
        "summaries": 0,
    }
    
    for v_dir in track(v_dirs, description="Analyzing Dataset..."):
        t_path = v_dir / "transcript.txt"
        c_path = v_dir / "comments.json"
        s_path = v_dir / "summary.txt"
        
        if t_path.exists():
            stats["transcripts"] += 1
            try:
                with open(t_path, "r", encoding="utf-8") as f:
                    stats["transcript_lines"] += sum(1 for _ in f)
            except Exception: pass
                
        if c_path.exists():
            stats["comments_files"] += 1
            try:
                data = json.loads(c_path.read_text(encoding="utf-8"))
                stats["total_comments"] += len(data.get("comments", []))
            except Exception:
                pass
                
        if s_path.exists():
            stats["summaries"] += 1
            
    # Print Report
    console.print("\n")
    table = Table(title="Dataset Extraction Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", justify="right", style="green", width=15)
    
    table.add_row("Total Video Folders", f"{stats['videos']:,}")
    table.add_section()
    table.add_row("Extracted Transcripts", f"{stats['transcripts']:,}")
    table.add_row("Total Transcript Lines", f"{stats['transcript_lines']:,}")
    table.add_section()
    table.add_row("Extracted Comment Files", f"{stats['comments_files']:,}")
    table.add_row("Total Individual Comments", f"{stats['total_comments']:,}")
    table.add_section()
    table.add_row("Generated Summaries", f"{stats['summaries']:,}")

    console.print(table)
    console.print(f"\n[bold]Transcript Completion Rate:[/bold] {(stats['transcripts']/stats['videos']*100 if stats['videos'] else 0):.1f}%")
    console.print(f"[bold]Comments Completion Rate:[/bold] {(stats['comments_files']/stats['videos']*100 if stats['videos'] else 0):.1f}%")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
