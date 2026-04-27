"""
tools/cleanup.py
----------------
Dataset maintenance tool to remove low-quality or incomplete data.
"""
import argparse
import shutil
import json
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn
from utils.helpers import parse_count

console = Console()

def cleanup(output_dir: Path, min_comments: int, dry_run: bool):
    if not output_dir.exists():
        console.print(f"[red]Error: {output_dir} not found.[/red]")
        return
    video_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    removed, freed = 0, 0

    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task("Cleaning up...", total=len(video_dirs))
        for v_dir in video_dirs:
            c_path = v_dir / "comments.json"
            reason = None
            if not c_path.exists(): reason = "Missing comments.json"
            else:
                try:
                    data = json.loads(c_path.read_text(encoding="utf-8"))
                    if len(data.get("comments", [])) < min_comments: reason = "Below threshold"
                except Exception: reason = "Corrupt"

            if reason:
                if not dry_run:
                    size = sum(f.stat().st_size for f in v_dir.glob('**/*') if f.is_file())
                    shutil.rmtree(v_dir, ignore_errors=True)
                    freed += size
                removed += 1
            progress.advance(task)

    console.print(f"\n[bold green]Cleanup Done![/bold green] {'(DRY RUN)' if dry_run else ''}")
    console.print(f" • Folders cleaned: {removed}")
    if not dry_run: console.print(f" • Space freed: {freed / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune low-quality datasets")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "output")
    parser.add_argument("--min-comments", type=parse_count, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cleanup(args.output_dir, args.min_comments, args.dry_run)
