"""
Category-Aware Video Link Maintenance Utility.

Cleans and validates video_pipeline/video.txt via:
1. --dedupe (Default): Removes duplicate URLs and empty category headers.
2. --verify: Fast OEmbed-based status check (Available, Private, Deleted).
3. --audit: Scans output/ directory to remove videos missing transcripts or comments.
"""
import argparse
import sys
import json
import urllib.request
import urllib.error
import shutil
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
    # Fallback if run standalone without parent path in sys.path
    import re
    def extract_video_id(url):
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else None

def clean_duplicates(file_path, dry_run=False):
    """Removes duplicate URLs and empty category headers."""
    path = Path(file_path)
    if not path.exists():
        console.print(f"[red][ERR] File not found: {file_path}[/red]")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    unique_lines = []
    seen_ids = set()
    removed_count = 0
    
    current_cat_header = None

    for line in lines:
        clean = line.strip()
        if not clean: continue

        if clean.startswith("# Category:"):
            current_cat_header = line
            continue
            
        if clean.startswith("#"):
            unique_lines.append(line)
            continue
            
        video_id = extract_video_id(clean)
        if not video_id:
            if current_cat_header:
                unique_lines.append(current_cat_header)
                current_cat_header = None
            unique_lines.append(line)
            continue
            
        if video_id in seen_ids:
            removed_count += 1
            continue
            
        seen_ids.add(video_id)
        if current_cat_header:
            unique_lines.append(current_cat_header)
            current_cat_header = None
        unique_lines.append(line)

    if removed_count == 0:
        console.print("[green]No duplicates found. The list is clean.[/green]")
        return

    if dry_run:
        console.print(f"[yellow][DRY RUN] Would remove {removed_count} duplicates.[/yellow]")
        return

    path.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
    console.print(f"[bold green]Success![/bold green] Removed {removed_count} duplicate link(s).")

def verify_links(file_path):
    """Fast validation via OEmbed API."""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    vids = [(l, extract_video_id(l)) for l in lines if extract_video_id(l) and not l.strip().startswith("#")]
    
    results = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
        task = progress.add_task("Verifying Status...", total=len(vids))
        for line, v_id in vids:
            try:
                oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={v_id}"
                with urllib.request.urlopen(oembed_url, timeout=5) as resp:
                    results.append({"id": v_id, "status": "OK", "info": json.loads(resp.read()).get("title", "Unknown")})
            except urllib.error.HTTPError as e:
                status = "PRIVATE" if e.code in (401, 403) else "DELETED"
                results.append({"id": v_id, "status": status, "info": f"HTTP {e.code}"})
            except Exception:
                results.append({"id": v_id, "status": "ERROR", "info": "Connection error"})
            progress.advance(task)

    table = Table(title="Verification Results")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Info")
    for r in results:
        if r['status'] != "OK":
            table.add_row(r['id'], r['status'], r['info'])
    console.print(table)

def audit_output(file_path, dry_run=False):
    """Removes 'dud' entries that failed extraction."""
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    kept_lines = []
    removed_count = 0
    
    for line in lines:
        v_id = extract_video_id(line.strip())
        if v_id and not line.strip().startswith("#"):
            v_dir = OUTPUT_DIR / v_id
            t_file = v_dir / "transcript.txt"
            c_file = v_dir / "comments.json"
            
            is_dud = not v_dir.exists() or not t_file.exists() or t_file.stat().st_size < 10
            if is_dud:
                removed_count += 1
                continue
        kept_lines.append(line)

    if removed_count == 0:
        console.print("[green]No duds found in extraction output.[/green]")
        return

    if not dry_run:
        path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        console.print(f"[bold green]Audited![/bold green] Removed {removed_count} duds.")
    else:
        console.print(f"[yellow][DRY RUN] Would remove {removed_count} duds.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Video Link Maintenance Master")
    parser.add_argument("--verify", action="store_true", help="Perform fast status check (OEmbed)")
    parser.add_argument("--audit", action="store_true", help="Remove duds based on output/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't apply changes")
    args = parser.parse_args()

    if args.verify:
        verify_links(VIDEO_FILE)
    elif args.audit:
        audit_output(VIDEO_FILE, args.dry_run)
    else:
        clean_duplicates(VIDEO_FILE, args.dry_run)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold red][HALT] Maintenance interrupted.[/bold red]")
        sys.exit(0)
