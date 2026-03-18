"""
Dataset Diversity Visualization Dashboard (Step 3).

Generates a visual summary of the discovered dataset, including:
- Category Distribution (Niches)
- Engagement Spread (View Tiers)
- Channel Diversity (Unique Channels)
- Content Coverage (Total Duration)

Supports terminal-based dashboards using `rich` and interactive premium JS dashboards 
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
import json
from datetime import datetime
import plotly.graph_objects as go
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

def generate_premium_dashboard(df, save_path=None):
    """Generates a high-fidelity, interactive HTML dashboard using Tailwind + Plotly Components."""
    console.print("\n[cyan]Generating premium dashboard...[/cyan]")
    
    # Pre-process data
    total_vids = len(df)
    unique_cats = df['category'].nunique()
    unique_channels = df['channel_title'].nunique()
    df['view_count_num'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
    df['duration_sec'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
    
    total_duration_raw = int(df['duration_sec'].sum())
    total_duration_fmt = fmt_duration(total_duration_raw)
    avg_views = int(df['view_count_num'].mean())
    
    # --- 1. Category Chart (Dynamic Height!) ---
    cat_counts = df['category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Count']
    cat_counts = cat_counts.sort_values(by='Count', ascending=True)
    # Ensure every category has at least 28 pixels of vertical space so they never squish
    cat_height = max(500, len(cat_counts) * 28)
    
    fig_cat = go.Figure(go.Bar(
        x=cat_counts['Count'].tolist(),
        y=cat_counts['Category'].tolist(),
        orientation='h',
        marker=dict(color='#06b6d4', opacity=0.8, line=dict(color='#0891b2', width=1))
    ))
    fig_cat.update_layout(
        height=cat_height, margin=dict(l=180, r=20, t=40, b=40),
        title=dict(text="Category Distribution", font=dict(color='white')), 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    cat_html = fig_cat.to_html(full_html=False, include_plotlyjs=False)
    
    # --- 2. Channel Chart (Top 15) ---
    chan_counts = df['channel_title'].value_counts().head(15).reset_index()
    chan_counts.columns = ['Channel', 'Count']
    chan_counts = chan_counts.sort_values(by='Count', ascending=True)
    
    fig_chan = go.Figure(go.Bar(
        x=chan_counts['Count'].tolist(),
        y=chan_counts['Channel'].tolist(),
        orientation='h',
        marker=dict(color='#d946ef', opacity=0.8, line=dict(color='#c026d3', width=1))
    ))
    fig_chan.update_layout(
        height=400, margin=dict(l=180, r=20, t=40, b=40),
        title=dict(text="Top 15 Channels", font=dict(color='white')), 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    chan_html = fig_chan.to_html(full_html=False, include_plotlyjs=False)
    
    fig_tier = go.Figure(go.Histogram(
        x=df['view_count_num'].tolist(),
        nbinsx=50,
        marker=dict(color='#10b981', opacity=0.8, line=dict(color='#059669', width=1))
    ))
    fig_tier.update_layout(
        height=400, margin=dict(l=60, r=20, t=40, b=40),
        title=dict(text="Engagement Spread (Views)", font=dict(color='white')), 
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    tier_html = fig_tier.to_html(full_html=False, include_plotlyjs=False)
    
    # --- 4. Interactive Scatter Plot ---
    valid_df = df[(df['duration_sec'] > 0) & (df['view_count_num'] > 0)]
    fig_scatter = go.Figure()
    
    for cat in valid_df['category'].unique():
        cat_df = valid_df[valid_df['category'] == cat]
        hover_text = cat_df.apply(lambda row: f"<b>{row['title']}</b><br>Channel: {row['channel_title']}<br>Views: {row['view_count_num']:,}<br>Duration (s): {row['duration_sec']}", axis=1).tolist()
        
        fig_scatter.add_trace(go.Scatter(
            x=cat_df['duration_sec'].tolist(), 
            y=cat_df['view_count_num'].tolist(), 
            mode='markers', 
            text=hover_text,
            hoverinfo='text',
            name=cat,
            marker=dict(opacity=0.7, size=8)
        ))
        
    fig_scatter.update_layout(
        height=700, margin=dict(l=80, r=20, t=60, b=60),
        title=dict(text="Views vs Duration (Double-Click Legend to Isolate)", font=dict(color='white')),
        template="plotly_dark",
        xaxis_title="Duration (Seconds)", yaxis_title="Views",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter", font_color="black")
    )
    scatter_html = fig_scatter.to_html(full_html=False, include_plotlyjs=False)

    # --- HTML Template (Tailwind + Plotly) ---
    html_template = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YTCBERT Diversity Dashboard</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly.js (Loaded ONCE) -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        body {{ font-family: 'Inter', sans-serif; }}
        .glass {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.1); }}
    </style>
</head>
<body class="bg-[#0f172a] text-slate-200 min-h-screen">
    
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 glass border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center gap-3">
            <div class="p-2 bg-cyan-500/20 rounded-lg text-cyan-400">
                <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            </div>
            <h1 class="text-xl font-bold tracking-tight text-white">YTCBERT <span class="text-slate-400 font-light">Diversity Dashboard</span></h1>
        </div>
        <div class="text-sm text-slate-400 bg-slate-800/50 px-3 py-1 rounded-full border border-slate-700">
            Internal Dataset Audit • {datetime.now().strftime("%Y-%m-%d")}
        </div>
    </nav>

    <main class="p-6 lg:p-10 space-y-10">
        
        <!-- Header / Metrics -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            <div class="glass p-6 rounded-2xl">
                <p class="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1">Total Videos</p>
                <h2 class="text-3xl font-bold text-white">{total_vids:,}</h2>
            </div>
            <div class="glass p-6 rounded-2xl">
                <p class="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1">Niches Coverage</p>
                <h2 class="text-3xl font-bold text-cyan-400">{unique_cats}</h2>
            </div>
            <div class="glass p-6 rounded-2xl">
                <p class="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1">Unique Channels</p>
                <h2 class="text-3xl font-bold text-magenta-400">{unique_channels}</h2>
            </div>
            <div class="glass p-6 rounded-2xl">
                <p class="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1">Avg Engagement</p>
                <h2 class="text-3xl font-bold text-green-400">{avg_views:,} <span class="text-sm font-normal text-slate-500">views</span></h2>
            </div>
             <div class="glass p-6 rounded-2xl">
                <p class="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-1">Total Duration</p>
                <h2 class="text-3xl font-bold text-amber-400">{total_duration_fmt}</h2>
            </div>
        </div>

        <!-- Major Visuals -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-10">
            
            <!-- Category Distribution: Needs massive vertical height, so place it in its own full-width block or a large column -->
            <div class="lg:col-span-12 glass p-4 rounded-3xl overflow-hidden">
                <div class="w-full overflow-y-auto" style="max-height: 800px;">
                    {cat_html}
                </div>
            </div>

            <!-- Side Panels -->
            <div class="lg:col-span-6 glass p-4 rounded-3xl">
                {chan_html}
            </div>

            <!-- Engagement Bar -->
            <div class="lg:col-span-6 glass p-4 rounded-3xl">
                {tier_html}
            </div>
            
            <!-- Scatter Plot -->
            <div class="lg:col-span-12 glass p-4 rounded-3xl">
                {scatter_html}
            </div>
        </div>

    </main>
    <footer class="p-10 text-center text-slate-500 text-sm">
        Generated by YTCBERT Engine Pipeline v1.0
    </footer>
</body>
</html>"""
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        console.print(f"[bold green]Premium dashboard explicitly saved to:[/bold green] {save_path}")
    else:
        fd, temp_path = tempfile.mkstemp(suffix=".html", prefix="ytcbert_dashboard_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_template)
        console.print(f"[bold green]Opening transient premium dashboard in browser...[/bold green] [dim]({temp_path})[/dim]")
        webbrowser.open(f"file://{temp_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize diversity and distributions of video.txt.")
    parser.add_argument("--file", default=BASE_DIR / "video.txt", help="Path to video list")
    parser.add_argument("--workers", type=int, default=15, help="Parallel workers for metadata fetching")
    parser.add_argument("--show-report", action="store_true", help="Open a premium interactive JS dashboard in browser (temporary)")
    parser.add_argument("--export-report", action="store_true", help="Save the premium interactive JS dashboard to project directory")
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
            
        generate_premium_dashboard(df, save_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red][HALT] Visualization interrupted by user.[/bold red]")
        sys.exit(0)
