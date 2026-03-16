"""
Dataset Diversity Visualization Dashboard (Step 3).

Generates a visual summary of the discovered dataset, including:
- Category Distribution (Niches)
- Engagement Spread (View Tiers)
- Channel Diversity (Unique Channels)
- Content Coverage (Total Duration & Word Counts)
"""

import os
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.align import Align
from dotenv import load_dotenv

# Ensure we can import from the engine directory
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from utils.helpers import extract_video_id, get_video_stats, fmt_duration

load_dotenv()
console = Console()

def parse_video_list(file_path):
    """Parses video.txt and returns a list of (url, category) tuples."""
    path = Path(file_path)
    if not path.exists():
        return []
    
    videos = []
    current_category = "Unknown"
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Category:"):
                current_category = line.split(":", 1)[1].strip()
            elif not line.startswith("#"):
                videos.append({"url": line, "category": current_category})
    
    return videos

def fetch_metadata(video_data, api_key):
    """Fetches full metadata for a video record."""
    v_id = extract_video_id(video_data['url'])
    if not v_id:
        return None
    
    stats = get_video_stats(v_id, api_key)
    if not stats:
        return None
    
    # Merge discover-time category with live stats
    record = {
        **video_data,
        **stats,
        "id": v_id
    }
    return record

def create_dashboard(df):
    """Generates the Rich-based visual dashboard."""
    
    # 1. Summary Statistics Panel
    stats_table = Table.grid(padding=1)
    stats_table.add_column(style="cyan", justify="right")
    stats_table.add_column(style="white")
    
    stats_table.add_row("Total Videos:", f"[bold]{len(df)}[/bold]")
    stats_table.add_row("Unique Categories:", str(df['category'].nunique()))
    stats_table.add_row("Unique Channels:", str(df['channel_title'].nunique()))
    stats_table.add_row("Avg Views:", f"{int(df['view_count'].mean()):,}")
    stats_table.add_row("Total Duration:", fmt_duration(df['duration'].sum()))

    # 2. Category Distribution (Bar Chart)
    cat_counts = df['category'].value_counts()
    cat_table = Table(title="Category Distribution", box=None, header_style="bold magenta")
    cat_table.add_column("Category")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Distribution")
    
    max_count = cat_counts.max()
    for cat, count in cat_counts.items():
        bar_len = int((count / max_count) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        cat_table.add_row(cat, str(count), f"[magenta]{bar}[/magenta] {count/len(df):.1%}")

    # 3. View Count Distribution (Engagement)
    bins = [0, 1000, 10000, 100000, 1000000, float('inf')]
    labels = ["<1k", "1k-10k", "10k-100k", "100k-1M", "1M+"]
    view_bins = pd.cut(df['view_count'], bins=bins, labels=labels).value_counts().reindex(labels)
    
    view_table = Table(title="Engagement Spread (Views)", box=None, header_style="bold green")
    view_table.add_column("Tier")
    view_table.add_column("Count", justify="right")
    view_table.add_column("Graph")
    
    max_view_bin = view_bins.max()
    for label, count in view_bins.items():
        count = count if not pd.isna(count) else 0
        bar_len = int((count / max_view_bin) * 15) if max_view_bin > 0 else 0
        bar = "█" * bar_len
        view_table.add_row(label, str(int(count)), f"[green]{bar}[/green]")

    # Assembly
    console.print(Align.center(Panel("[bold yellow]YTCBERT Dataset Diversity Dashboard[/bold yellow]", expand=False, border_style="yellow")))
    
    col1 = Panel(stats_table, title="[bold cyan]Key Metrics[/bold cyan]", border_style="cyan")
    col2 = Panel(view_table, title="[bold green]Engagement Distribution[/bold green]", border_style="green")
    
    console.print(Columns([col1, col2], expand=True))
    console.print(Panel(cat_table, border_style="magenta"))

def main():
    parser = argparse.ArgumentParser(description="Visualize diversity and distributions of video.txt.")
    parser.add_argument("--file", default=BASE_DIR / "video.txt", help="Path to video list")
    parser.add_argument("--workers", type=int, default=15, help="Parallel workers for metadata fetching")
    args = parser.parse_args()

    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        console.print("[red][ERR] YOUTUBE_API_KEY is required for detailed visuals (views/duration).[/red]")
        return

    # 1. Parse file
    video_list = parse_video_list(args.file)
    if not video_list:
        console.print(f"[yellow]No videos found in {args.file}[/yellow]")
        return

    # 2. Fetch Metadata in Parallel
    records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Fetching metadata for visuals...", total=len(video_list))
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch_metadata, v, api_key) for v in video_list]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    records.append(res)
                progress.advance(task)

    if not records:
        console.print("[red]Failed to fetch metadata for any videos.[/red]")
        return

    # 3. Analyze and Visualize
    df = pd.DataFrame(records)
    create_dashboard(df)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red][HALT] Visualization interrupted by user.[/bold red]")
        sys.exit(0)
