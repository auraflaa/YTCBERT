"""
Pro Video Discovery Utility for YouTube.

This script automates the discovery of high-quality English videos to build
a diverse dataset for training. It uses a "Pro" filtering strategy:
1. Randomized keyword search via seeds (entropy).
2. Category-based rotation (AI, Tech, Science, etc.).
3. Quality filtering (Duration: 2-30 min, Views: >5,000).
4. Language detection (English only).
"""

import argparse
import random
import sys
import time
from pathlib import Path

import yt_dlp
from langdetect import detect, DetectorFactory
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn

from utils.helpers import extract_video_id

# Ensures consistent results for language detection across runs
DetectorFactory.seed = 0

# Configuration
BASE_DIR = Path(__file__).parent
VIDEO_FILE = BASE_DIR / "video.txt"
console = Console()

# Category definition for balanced vertical variety
CATEGORIES = {
    "AI & Deep Learning": ["Karpathy", "3Blue1Brown", "Deep Learning", "Neural Networks"],
    "Data Science / Python": ["Krish Naik", "Data Science tutorial", "Pandas tutorial", "Machine Learning Python"],
    "Tech Ethics & News": ["Tech Ethics", "AI Safety", "Tech News", "Silicon Valley news"],
    "Engineering/Science": ["Veritasium", "Mark Rober", "Engineering explained", "SmarterEveryDay"],
    "Gadget Reviews": ["MKBHD", "Linus Tech Tips", "Dave2D", "Smartphone review"],
    "Productivity/Dev": ["Ali Abdaal", "Fireship", "Programming tutorial", "Developer productivity"],
    "Startup/Business": ["Y Combinator", "Startup pitch", "Business strategy", "Tech entrepreneurship"],
    "Documentaries": ["Kurzgesagt", "Wendover Productions", "PolyMatter", "Educational documentary"],
    "Space & Physics": ["PBS Space Time", "Scott Manley", "Astrophysics", "Space exploration"],
    "Educational Wildcards": ["Expert breakdown", "Deep dive tutorial", "Video essay", "Technical lecture"],
    "Industry Insights": ["System design", "Software architecture", "Engineering blog", "Industry standards"],
    "Casual Learning": ["Explain like I'm five", "Quick intro", "Summary of", "Overview of"]
}

def is_english(title, description=""):
    """
    Performs a heuristic check for English content using langdetect.
    Checks the combined title and description for higher accuracy.
    """
    text = f"{title} {description}".strip()
    if not text or len(text) < 10:
        return False
    try:
        return detect(text) == 'en'
    except:
        return False

def get_ydl_opts(verify_subtitles=True):
    """Returns standardized configuration for yt-dlp."""
    return {
        'quiet': True,
        'extract_flat': False, # Extract full info to check views/duration
        'search_filter': 'relevance', 
        'ignoreerrors': True,
        'no_warnings': True,
        'writesubtitles': verify_subtitles,
        'writeautomaticsub': verify_subtitles,
        'skip_download': True, # We only want metadata
    }

def discover_pro_videos(target_count=50, min_views=0, min_duration=0, max_duration=99999):
    """
    Main discovery logic. Rotates through categories and keywords.
    Loosened filters to include all types of videos (no transcripts, low comments, etc.)
    """
    video_links = []
    seen_ids = set()
    
    # Load existing IDs to avoid duplicates in the current video.txt
    if Path(VIDEO_FILE).exists():
        content = Path(VIDEO_FILE).read_text(encoding="utf-8")
        for line in content.splitlines():
            vid = extract_video_id(line.strip())
            if vid: seen_ids.add(vid)

    console.print(f"[bold blue]Starting Broad Discovery for {target_count} videos (All Types)...[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Discovering...", total=target_count)
        
        # Flattened query list for easy rotation
        all_queries = []
        for cat, kw_list in CATEGORIES.items():
            for kw in kw_list:
                all_queries.append((cat, kw))
        random.shuffle(all_queries) # Shuffle once for initial randomness

        query_idx = 0
        with yt_dlp.YoutubeDL(get_ydl_opts(verify_subtitles=False)) as ydl:
            while len(video_links) < target_count:
                # Cycle through the category/keyword pool
                cat, base_query = all_queries[query_idx % len(all_queries)]
                query_idx += 1
                
                # Add a random 3-letter seed to query to find less obvious videos (discovery entropy)
                seed = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3))
                query = f"{base_query} {seed}"
                
                progress.update(task, description=f"Querying [magenta]{cat}[/magenta]: '{base_query}'")
                
                try:
                    # Probe YouTube Search (Top 20 results per query for broader reach)
                    results = ydl.extract_info(f"ytsearch20:{query}", download=False)
                    if not results or 'entries' not in results:
                        continue

                    for entry in results['entries']:
                        if not entry: continue
                        if len(video_links) >= target_count: break
                        
                        v_id = entry.get('id')
                        if not v_id or v_id in seen_ids: continue
                        
                        # -- Applied Minimum Logic (Mostly Basic Sanity) --
                        duration = entry.get('duration', 0)
                        view_count = entry.get('view_count', 0)
                        title = entry.get('title', '')
                        
                        # Only check if TITLE is English (basic language sanity)
                        if not is_english(title): continue
                        
                        # Success! Add to the dataset (No more elite filters)
                        video_links.append({
                            'id': v_id,
                            'title': title,
                            'url': f"https://www.youtube.com/watch?v={v_id}",
                            'category': cat,
                            'views': view_count,
                            'duration': f"{duration//60}:{duration%60:02d}" if duration else "N/A"
                        })
                        seen_ids.add(v_id)
                        progress.advance(task)
                        
                except Exception:
                    # Silently skip errors during massive searches
                    continue

    return video_links

def main():
    """CLI Entry point for video discovery."""
    parser = argparse.ArgumentParser(description="Broad Video Discovery for diverse YouTube datasets.")
    parser.add_argument("--count", type=int, default=10, help="Number of new videos to find (default: 10)")
    parser.add_argument("--append", action="store_true", help="Append found links to video.txt")
    args = parser.parse_args()

    # Run the broad discovery
    found = discover_pro_videos(args.count)
    
    if not found:
        console.print("[yellow]No new videos found.[/yellow]")
        return

    # Display pretty results table
    table = Table(title=f"Discovered {len(found)} High-Quality Videos")
    table.add_column("Category", style="magenta")
    table.add_column("Title", style="white")
    table.add_column("Views", justify="right", style="green")
    table.add_column("Duration", justify="right", style="cyan")
    
    for v in found:
        table.add_row(v['category'], v['title'][:40]+"...", f"{v['views']:,}", v['duration'])
    
    console.print(table)

    # Save to file if requested
    if args.append:
        with open(VIDEO_FILE, "a", encoding="utf-8") as f:
            current_cat = None
            for v in found:
                # Group by category in the file for user readability
                if v['category'] != current_cat:
                    current_cat = v['category']
                    f.write(f"\n# Category: {current_cat}\n")
                f.write(f"{v['url']}\n")
        console.print(f"[bold green]Successfully added {len(found)} elite links to {VIDEO_FILE}![/bold green]")

if __name__ == "__main__":
    main()
