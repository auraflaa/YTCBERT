"""
tools/maintenance/cleanup_dataset.py
------------------------------------
Dataset maintenance tool to remove low-quality or incomplete data.
"""

import argparse
import shutil
import json
import sys
import os
from pathlib import Path

# --- ENSURE ROOT MODULES ARE FINDABLE ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn

# Local Imports
try:
    from utils.helpers import parse_count
except ImportError:
    # Minimal fallback
    def parse_count(x): return int(x)

console = Console()

def cleanup(output_dir: Path, min_comments: int, dry_run: bool):
    if not output_dir.exists():
        console.print(f"[red]Error: {output_dir} not found.[/red]")
        return

    video_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    total = len(video_dirs)
    removed = 0
    size_freed = 0
    
    console.print(f"\n[bold blue]Scanning {total} folders for cleanup...[/bold blue]\n")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing cleanup...", total=total)
        for v_dir in video_dirs:
            c_path = v_dir / "comments.json"
            delete_reason = None
            if not c_path.exists():
                delete_reason = "Missing comments.json"
            else:
                try:
                    data = json.loads(c_path.read_text(encoding="utf-8"))
                    count = len(data.get("comments", []))
                    if count < min_comments:
                        delete_reason = f"Comments {count} < {min_comments}"
                except Exception:
                    delete_reason = "Corrupt comments.json"

            if delete_reason:
                if not dry_run:
                    size = sum(f.stat().st_size for f in v_dir.glob('**/*') if f.is_file())
                    shutil.rmtree(v_dir, ignore_errors=True)
                    removed += 1
                    size_freed += size
                else:
                    removed += 1
            progress.advance(task)

    action = "found" if dry_run else "removed"
    console.print(f"\n[bold green]Cleanup Complete![/bold green]")
    console.print(f" • Folders {action}: [yellow]{removed}[/yellow]")
    if not dry_run:
        console.print(f" • Disk space freed: [yellow]{size_freed / 1024 / 1024:.2f} MB[/yellow]")
    console.print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up low-quality dataset folders.")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "output", help="Path to output directory")
    parser.add_argument("--min-comments", type=str, default="1", help="Min comments to keep a folder")
    parser.add_argument("--dry-run", action="store_true", help="Don't delete, just report")
    args = parser.parse_args()

    # Resolve threshold
    try:
        from utils.helpers import parse_count
        min_c = parse_count(args.min_comments)
    except Exception:
        min_c = int(args.min_comments)

    cleanup(args.output_dir, min_c, args.dry_run)
