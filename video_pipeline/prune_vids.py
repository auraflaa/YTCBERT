"""
Thin-Data Purge Utility (Step 1.5).

Scans the extraction output directory and identifies "dud" videos:
- Missing transcripts
- Missing or empty comments.json
- Missing output folders
"""
import argparse
import sys
import shutil
import datetime
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

# Engine-relative imports
BASE_DIR = Path(__file__).parent
VIDEO_FILE = BASE_DIR / "video.txt"
OUTPUT_DIR = Path("output")
console = Console()

try:
    from utils.helpers import extract_video_id
except ImportError:
    import re
    def extract_video_id(url):
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

def audit_video(v_id):
    """Checks if a video has actually yielded usable data."""
    v_dir = OUTPUT_DIR / v_id
    if not v_dir.exists():
        return False, "Output folder missing"
    
    transcript_file = v_dir / "transcript.txt"
    comments_file = v_dir / "comments.json"
    
    if not transcript_file.exists() or transcript_file.stat().st_size < 10:
        return False, "Missing/Empty Transcript"
    if not comments_file.exists() or comments_file.stat().st_size < 10:
        return False, "Missing/Empty Comments"
        
    return True, "OK"

def main():
    parser = argparse.ArgumentParser(description="Prune 'dud' videos from video.txt based on extraction results.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without changing files")
    args = parser.parse_args()

    if not VIDEO_FILE.exists():
        console.print(f"[red][ERR] {VIDEO_FILE} not found.[/red]")
        return

    raw_lines = VIDEO_FILE.read_text(encoding="utf-8").splitlines()
    kept_lines = []
    removed_count = 0
    reasons = {}

    console.print(f"[bold blue]Auditing extraction data in ./{OUTPUT_DIR}...[/bold blue]\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task("Auditing...", total=len(raw_lines))
        for line in raw_lines:
            v_id = extract_video_id(line.strip())
            if v_id and not line.strip().startswith("#"):
                is_valid, reason = audit_video(v_id)
                if is_valid:
                    kept_lines.append(line)
                else:
                    removed_count += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
            else:
                kept_lines.append(line)
            progress.advance(task)

    if removed_count == 0:
        console.print("\n[bold green]✅ No duds found! Your dataset is clean.[/bold green]")
        return

    table = Table(title="Pruning Summary")
    table.add_column("Reason", style="red")
    table.add_column("Count", justify="right")
    for r, count in reasons.items():
        table.add_row(r, str(count))
    console.print(table)

    if not args.dry_run:
        master_backup_dir = BASE_DIR.parent / "backups"
        master_backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = master_backup_dir / f"video_prune_{timestamp}.bak"
        
        shutil.copy(VIDEO_FILE, backup_path)
        VIDEO_FILE.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        console.print(f"\n[bold green]Success![/bold green] Results:")
        console.print(f" - Backup created: [dim]{backup_path.name}[/dim]")
        console.print(f" - Removed {removed_count} dead links.")
    else:
        console.print(f"\n[yellow][DRY RUN] Would have removed {removed_count} entries.[/yellow]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red][HALT] Pruning interrupted.[/bold red]")
        sys.exit(0)
