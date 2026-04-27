"""
tools/discovery/discover.py
--------------------------
Broad Video Discovery for diverse YouTube datasets.
"""

import argparse
import random
import sys
import time
import os
from pathlib import Path

# Add project root to sys.path so we can import utils
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn, ProgressColumn
from collections import deque
from utils.helpers import extract_video_id, parse_count, resolve_data_path

from rich.text import Text
from datetime import timedelta

class DiscoveryETAColumn(ProgressColumn):
    """Deeply smoothed ETA column using a 60-second time-based window."""
    def __init__(self, time_window=60):
        self.time_window = time_window
        self.history = deque() # Stores (timestamp, completed_count)
        super().__init__()

    def update_history(self, completed_count):
        now = time.time()
        self.history.append((now, completed_count))
        while self.history and self.history[0][0] < (now - self.time_window):
            self.history.popleft()

    def render(self, task):
        if not task.total or len(self.history) < 2:
            return Text("-:--:--", style="dim")
        first_time, first_count = self.history[0]
        last_time, last_count = self.history[-1]
        delta_time = last_time - first_time
        delta_items = last_count - first_count
        if delta_time < 5 or delta_items <= 0:
            if task.elapsed > 0 and task.completed > 0:
                items_per_sec = task.completed / task.elapsed
            else:
                return Text("-:--:--", style="dim")
        else:
            items_per_sec = delta_items / delta_time
        remaining_count = task.total - task.completed
        remaining_seconds = (remaining_count / items_per_sec) if items_per_sec > 0 else 0
        total_estimated_seconds = task.elapsed + remaining_seconds
        total_m, total_s = divmod(int(total_estimated_seconds), 60)
        total_h, total_m = divmod(total_m, 60)
        total_str = f"{total_h}h {total_m}m" if total_h > 0 else f"{total_m}m {total_s}s"
        from datetime import datetime
        finish_time = (datetime.now() + timedelta(seconds=int(remaining_seconds))).strftime("%H:%M")
        res = Text()
        res.append("Tot:", style="dim")
        res.append(total_str, style="yellow")
        res.append(" | ", style="dim")
        res.append("ETA:", style="dim")
        res.append(finish_time, style="bold green")
        return res

# Configuration
DATA_DIR = root_dir / "data"
DATA_DIR.mkdir(exist_ok=True)
VIDEO_FILE = resolve_data_path("video.txt", base_dir=root_dir)
console = Console()

# Category definitions
CATEGORIES = {
    "AI & Deep Learning": ["Karpathy", "Neural Networks", "LLM architecture", "Transformer models"],
    "Data Science / Python": ["Krish Naik", "Scikit-learn tutorial", "Data visualization", "SQL for Data Science"],
    "Tech Ethics & News": ["AI Alignment", "Privacy laws tech", "Silicon Valley news", "Tech monopoly debate"],
    "Software Engineering": ["System Design", "Microservices architecture", "Clean Code principles", "LeetCode solutions"],
    "Cybersecurity": ["Penetration testing", "Zero trust architecture", "Cyber attack analysis", "Encryption explained"],
    "Cloud Computing": ["AWS vs Azure vs GCP", "Docker and Kubernetes", "Serverless architecture", "Cloud migration"],
    "Physics & Space": ["PBS Space Time", "Astrophysics", "Quantum mechanics", "James Webb Telescope"],
    "Biology & Medicine": ["Molecular biology", "Genetics explained", "Medical breakthroughs", "Neuroscience"],
    "General Engineering": ["Structural engineering", "Mechanical design", "Mark Rober", "Veritasium"],
    "Mathematics": ["Number Theory", "Linear Algebra", "Calculus visual", "3Blue1Brown"],
    "Business": ["Macroeconomics", "Startups & VC", "Property investment", "Global inflation"],
    "Lifestyle": ["Interior Design", "Cooking & Food", "Fitness & Health", "Travel documentary"]
}

def is_english(title, description=""):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        text = f"{title} {description}".strip()
        if not text or len(text) < 10: return False
        return detect(text) == 'en'
    except Exception: return True

def get_cat_style(category):
    idx = sum(ord(c) for c in category) % 8
    colors = ["blue", "red", "bright_blue", "orange3", "purple", "deep_pink", "hot_pink", "chartreuse4"]
    return colors[idx]

def get_ydl_opts(verify_subtitles=True, cookies_path=None):
    opts = {
        'quiet': True, 'extract_flat': False, 'search_filter': 'relevance', 
        'ignoreerrors': False, 'no_warnings': True, 'writesubtitles': verify_subtitles,
        'writeautomaticsub': verify_subtitles, 'skip_download': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    if cookies_path and Path(cookies_path).exists(): opts['cookiefile'] = str(cookies_path)
    return opts

def count_existing_categories():
    counts = {cat: 0 for cat in CATEGORIES.keys()}
    if not Path(VIDEO_FILE).exists(): return counts
    current_cat = None
    with open(VIDEO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Category:"):
                current_cat = line.split(":", 1)[1].strip()
            elif line and not line.startswith("#"):
                if current_cat in counts: counts[current_cat] += 1
                else: counts[current_cat] = counts.get(current_cat, 0) + 1
    return counts

def discover_pro_videos(target_count=50, max_per_channel=20, min_comments=10, min_length_mins=1.0, cookies=None):
    video_links, seen_ids, channel_counts = [], set(), {}
    rejections = {"Shorts": 0, "Low Comments": 0, "Duplicate": 0, "Blocked/Private": 0, "Other": 0}
    
    existing_stats = count_existing_categories()
    total_existing = sum(existing_stats.values())
    
    if Path(VIDEO_FILE).exists():
        content = Path(VIDEO_FILE).read_text(encoding="utf-8")
        for line in content.splitlines():
            vid = extract_video_id(line.strip())
            if vid: seen_ids.add(vid)

    if total_existing >= target_count:
        console.print(f"[yellow]Target reached! You already have {total_existing}/{target_count} videos.[/yellow]")
        return []

    remaining = target_count - total_existing
    console.print(f"[bold blue]Discovery Mode[/bold blue] | Target: {target_count} | Existing: {total_existing} | Needs: {remaining}\n")
    
    eta_col = DiscoveryETAColumn(time_window=60)
    with Progress(SpinnerColumn(), TextColumn("{task.description}", markup=True), BarColumn(bar_width=None),
                  MofNCompleteColumn(), TaskProgressColumn(), TextColumn("•"), eta_col,
                  console=console, refresh_per_second=4, expand=True) as progress:
        task = progress.add_task("Discovering...", total=target_count, completed=total_existing)
        query_pool = [{"cat": c, "kw": k} for c, kws in CATEGORIES.items() for k in kws]
        random.shuffle(query_pool)
        
        try: import yt_dlp
        except ImportError: return []

        with yt_dlp.YoutubeDL(get_ydl_opts(verify_subtitles=False, cookies_path=cookies)) as ydl:
            while (total_existing + len(video_links)) < target_count:
                current_balance = {cat: existing_stats.get(cat, 0) for cat in CATEGORIES}
                for v in video_links: current_balance[v['category']] += 1
                min_count = min(current_balance.values())
                target_cat = random.choice([cat for cat, c in current_balance.items() if c == min_count])
                query_obj = random.choice([q for q in query_pool if q['cat'] == target_cat])
                query = f"{query_obj['kw']} {random.choice(['lecture', 'explained', 'tutorial', 'guide'])}"
                
                cat_style = get_cat_style(target_cat)
                progress.update(task, description=f"[{cat_style}]{target_cat:14}[/{cat_style}]")

                try:
                    results = ydl.extract_info(f"ytsearch30:{query}", download=False)
                    if not results or 'entries' not in results: continue
                    time.sleep(random.uniform(1.0, 3.0))
                    
                    for entry in results['entries']:
                        if not entry or (total_existing + len(video_links)) >= target_count: break
                        v_id = entry.get('id')
                        if not v_id or v_id in seen_ids: continue
                        if entry.get('availability', 'public') != 'public' or entry.get('is_private'): continue
                        
                        uploader = entry.get('uploader_id') or "Unknown"
                        if channel_counts.get(uploader, 0) >= max_per_channel: continue
                        if not is_english(entry.get('title', '')): continue
                        
                        duration = entry.get('duration', 0) or 0
                        comment_count = entry.get('comment_count')
                        if duration < (min_length_mins * 60): continue
                        if min_comments > 0 and (comment_count is None or comment_count < min_comments): continue

                        video_obj = {'id': v_id, 'title': entry.get('title'), 'url': f"https://www.youtube.com/watch?v={v_id}", 'category': target_cat, 'views': entry.get('view_count', 0), 'duration': f"{duration//60}:{duration%60:02d}"}
                        video_links.append(video_obj)
                        seen_ids.add(v_id)
                        channel_counts[uploader] = channel_counts.get(uploader, 0) + 1
                        
                        with open(VIDEO_FILE, "a", encoding="utf-8") as f:
                            f.write(f"\n# Category: {target_cat}\n{video_obj['url']}\n")
                        progress.advance(task)
                        eta_col.update_history(progress.tasks[task].completed)
                except Exception: continue
    return video_links

def main():
    parser = argparse.ArgumentParser(description="YouTube Video Discovery")
    parser.add_argument("--count", type=parse_count, default=10)
    parser.add_argument("--cookies", type=str, default=None)
    parser.add_argument("--min-comments", type=int, default=10)
    args = parser.parse_args()
    
    # Resolve cookies path relative to project root
    c_path = resolve_data_path(args.cookies, base_dir=root_dir) if args.cookies else None
    
    found = discover_pro_videos(args.count, cookies=c_path, min_comments=args.min_comments)
    if found:
        table = Table(title=f"Discovered {len(found)} Videos")
        table.add_column("Category", style="magenta")
        table.add_column("Title", style="white", overflow="ellipsis", max_width=50)
        table.add_column("Views", justify="right", style="green")
        for v in found: table.add_row(v['category'], v['title'], f"{v['views']:,}")
        console.print(table)

if __name__ == "__main__":
    main()
