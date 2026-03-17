"""
Dataset Diversity Visualization Dashboard (Step 3).

Generates a visual summary of the discovered dataset, including:
- Category Distribution (Niches)
- Engagement Spread (View Tiers)
- Channel Diversity (Unique Channels)
- Content Coverage (Total Duration)

Supports terminal-based dashboards using `rich` and interactive HTML dashboards using `plotly` 
(--show-report / --export-report).
"""

import os
import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TaskProgressColumn
from rich.align import Align
from dotenv import load_dotenv

# Ensure we can import from the engine directory
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

from utils.helpers import extract_video_id, get_video_stats, get_video_stats_batch, fmt_duration

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

def fetch_metadata_batch(video_data_batch, api_key):
    """Fetches metadata for a batch of up to 50 videos using a single API request."""
    id_to_data = {}
    valid_ids = []
    for data in video_data_batch:
        v_id = extract_video_id(data['url'])
        if v_id:
            id_to_data[v_id] = data
            valid_ids.append(v_id)
            
    if not valid_ids:
        return []
        
    stats_dict = get_video_stats_batch(valid_ids, api_key)
    
    records = []
    # Merge discover-time category with live stats
    for v_id, stats in stats_dict.items():
        if stats:
            records.append({
                **id_to_data[v_id],
                **stats,
                "id": v_id
            })
    return records

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

def generate_plotly_report(df, save_path=None):
    """Generates an interactive HTML dashboard using Plotly."""
    console.print("\n[cyan]Generating interactive Plotly dashboard...[/cyan]")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Most Common Categories (Sorted)", "Top 10 Channels by Content", 
                        "Views Distribution", "Views vs Duration (Seconds)"),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # 1. Category Chart (Horizontal Bar Chart for readability, sorted low to high so largest is at top)
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    cat_counts = cat_counts.sort_values(by='Count', ascending=True)
    fig.add_trace(go.Bar(
        y=cat_counts['Category'], 
        x=cat_counts['Count'], 
        orientation='h',
        marker=dict(color='cyan'),
        name="Category",
        showlegend=False
    ), row=1, col=1)
    
    # 2. Top Channels Bar Chart
    channel_counts = df['channel_title'].value_counts().head(10).reset_index()
    channel_counts.columns = ['Channel', 'Count']
    channel_counts = channel_counts.sort_values(by='Count', ascending=True)
    fig.add_trace(go.Bar(
        y=channel_counts['Channel'], 
        x=channel_counts['Count'], 
        orientation='h',
        marker=dict(color='magenta'),
        name="Channels",
        showlegend=False
    ), row=1, col=2)
    
    # 3. Views Histogram
    # Ensure view_count is numeric
    df['view_count_num'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
    
    fig.add_trace(go.Histogram(
        x=df['view_count_num'], 
        nbinsx=40, 
        marker=dict(color='green'),
        name="All Views",
        showlegend=False
    ), row=2, col=1)
                  
    # 4. Views vs Duration Scatter
    # IMPORTANT: df['duration'] from get_video_stats is already an integer (total seconds).
    df['duration_sec'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
    
    # Only scatter valid durations/views
    valid_df = df[(df['duration_sec'] > 0) & (df['view_count_num'] > 0)]
    
    # Enable segment extraction by splitting the scatter into Category traces!
    # This automatically builds an interactive filter Legend on the right side.
    unique_cats = valid_df['category'].unique()
    for cat in unique_cats:
        cat_df = valid_df[valid_df['category'] == cat]
        hover_text = cat_df.apply(lambda row: f"<b>{row['title']}</b><br>Channel: {row['channel_title']}<br>Views: {row['view_count_num']:,}<br>Duration (s): {row['duration_sec']}", axis=1)
        
        fig.add_trace(go.Scatter(
            x=cat_df['duration_sec'], 
            y=cat_df['view_count_num'], 
            mode='markers', 
            text=hover_text,
            hoverinfo='text',
            name=cat,          # Adds to Legend!
            marker=dict(opacity=0.7, size=8)
        ), row=2, col=2)
                  
    fig.update_layout(
        height=900, 
        title_text="<b>YTCBERT Dataset Diversity Report</b>", 
        title_x=0.5,
        showlegend=True,  # Turn on the legend for the scatter plot
        legend_title_text="Isolate Segments (Double-Click):",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial", font_color="black")
    )
    
    # Update axes titles for clarity
    fig.update_xaxes(title_text="Video Count", row=1, col=1)
    fig.update_xaxes(title_text="Video Count", row=1, col=2)
    fig.update_xaxes(title_text="Views", row=2, col=1)
    fig.update_xaxes(title_text="Duration (Seconds)", row=2, col=2)
    
    fig.update_yaxes(title_text="Category", row=1, col=1)
    fig.update_yaxes(title_text="Channel", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_yaxes(title_text="Views", row=2, col=2)
    
    html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[bold green]Report explicitly saved to:[/bold green] {save_path}")
    else:
        fd, temp_path = tempfile.mkstemp(suffix=".html", prefix="ytcbert_report_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_content)
        console.print(f"[bold green]Opening transient report in browser...[/bold green] [dim]({temp_path})[/dim]")
        webbrowser.open(f"file://{temp_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize diversity and distributions of video.txt.")
    parser.add_argument("--file", default=BASE_DIR / "video.txt", help="Path to video list")
    parser.add_argument("--workers", type=int, default=15, help="Parallel workers for metadata fetching")
    parser.add_argument("--show-report", action="store_true", help="Open an interactive Plotly HTML report in browser (temporary)")
    parser.add_argument("--export-report", action="store_true", help="Save the interactive Plotly HTML report to project directory")
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
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
        expand=True,
        refresh_per_second=4
    ) as progress:
        task = progress.add_task("[cyan]Fetching metadata for visuals...", total=len(video_list))
        
        # Batch requests (max 50 per YouTube Data API limits)
        batches = [video_list[i:i + 50] for i in range(0, len(video_list), 50)]
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_batchlen = {
                executor.submit(fetch_metadata_batch, b, api_key): len(b) 
                for b in batches
            }
            try:
                for future in as_completed(future_to_batchlen):
                    batch_len = future_to_batchlen[future]
                    try:
                        res_batch = future.result()
                        if res_batch:
                            records.extend(res_batch)
                        progress.advance(task, batch_len)
                    except RuntimeError as e:
                        if str(e) == "YOUTUBE_QUOTA_EXCEEDED":
                            # Terminate all threads and raise a special flag
                            for f in future_to_batchlen.keys():
                                f.cancel()
                            raise RuntimeError("QUOTA_HALT")
                        else:
                            progress.advance(task, batch_len)
            except KeyboardInterrupt:
                progress.console.print("\n[bold red][HALT] Visualization interrupted by user.[/bold red]")
                for f in future_to_batchlen.keys():
                    f.cancel()
                pass
            except RuntimeError as e:
                if str(e) == "QUOTA_HALT":
                    progress.console.print(Panel(
                        "[bold red][BLOCK] YOUTUBE DATA API QUOTA EXCEEDED.[/bold red]\n\n"
                        "Your provided YOUTUBE_API_KEY has exhausted its daily limit of 10,000 units.\n"
                        "The quota resets daily at midnight Pacific Time (PT).\n"
                        "Please use a different key or wait until the reset to continue visualization.",
                        title="YouTube API Halt", border_style="red"
                    ))
                    # Wait for progress bar to cleanly exit, then return early
                    return
                raise

    if not records:
        console.print("[red]Failed to fetch metadata for any videos.[/red]")
        return

    # 3. Analyze and Visualize (Terminal)
    df = pd.DataFrame(records)
    create_dashboard(df)
    
    # 4. Interactive Report (If Requested)
    if args.show_report or args.export_report:
        if args.export_report:
            reports_dir = BASE_DIR.parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            save_path = reports_dir / "diversity_report.html"
        else:
            save_path = None
            
        generate_plotly_report(df, save_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red][HALT] Visualization interrupted by user.[/bold red]")
        sys.exit(0)
