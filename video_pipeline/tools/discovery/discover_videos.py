"""
tools/discovery/discover_videos.py
----------------------------------
Broad Video Discovery for diverse YouTube datasets.
"""

import argparse
import random
import sys
import time
import os
from pathlib import Path
from collections import deque
from datetime import timedelta

# --- ENSURE ROOT MODULES ARE FINDABLE ---
# This allows the script to be run from anywhere while still finding 'utils'
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, ProgressColumn
from rich.text import Text

# Local Imports
try:
    from utils.helpers import extract_video_id, parse_count, resolve_data_path
except ImportError:
    # Fallback for manual debug
    print(f"Error: Could not find 'utils' module. ROOT_DIR detected as: {ROOT_DIR}")
    sys.exit(1)

# Configuration
VIDEO_FILE = resolve_data_path("video.txt")
console = Console()

class DiscoveryETAColumn(ProgressColumn):
    """Deeply smoothed ETA column using a 60-second time-based window."""
    def __init__(self, time_window=60):
        self.time_window = time_window
        self.history = deque() 
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
        
        if total_h > 0:
            total_str = f"{total_h}h {total_m}m"
        else:
            total_str = f"{total_m}m {total_s}s"
        
        from datetime import datetime
        finish_time = (datetime.now() + timedelta(seconds=int(remaining_seconds))).strftime("%H:%M")
        
        res = Text()
        res.append("Tot:", style="dim")
        res.append(total_str, style="yellow")
        res.append(" | ", style="dim")
        res.append("ETA:", style="dim")
        res.append(finish_time, style="bold green")
        return res

# [The rest of the logic remains the same as discover_videos.py]
# (Categories, is_english, count_existing_categories, discover_pro_videos, main)

CATEGORIES = {
    # --- TECH & DATA (The Core) ---
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
    "Chemistry": ["Chemical reactions", "Organic chemistry", "Material science", "Periodic table deep dive"],
    "Personal Finance": ["Investment strategies", "Index funds explained", "Credit score optimization", "Retirement planning"],
    "Macroeconomics": ["Global inflation", "Central banks", "Supply chain crisis", "Economic history"],
    "Cryptocurrency/Web3": ["Blockchain technology", "Ethereum smart contracts", "Bitcoin news", "DeFi explained"],
    "Startups & VC": ["Y Combinator", "Pitch deck analysis", "Venture capital trends", "SaaS business model"],
    "Real Estate": ["Property investment", "Housing market analysis", "Commercial real estate", "House flipping"],
    "Travel & Tourism": ["Luxury travel", "Budget backpacking", "Hidden gems Japan", "Travel documentary"],
    "Fashion & Style": ["Streetwear trends", "Luxury brand history", "Sustainable fashion", "Watch collecting"],
    "Interior Design": ["Minimalist home decor", "Architectural Digest", "Small apartment ideas", "DIY home renovation"],
    "Cooking & Food": ["Gordon Ramsay", "Street food tour", "Molecular gastronomy", "Traditional Italian recipes"],
    "Fitness & Health": ["Strength training", "Longevity science", "Nutrition myths", "Marathon preparation"],
    "Film & Cinema": ["Video essay movies", "Cinematography analysis", "Film history", "Screenwriting tips"],
    "Music Theory": ["Jazz improvisation", "Music production tutorial", "Synthesizer history", "Musicology"],
    "Gaming Culture": ["Esports industry", "Game design analysis", "Retro gaming", "Speedrunning documentary"],
    "Literature & Books": ["Classic literature", "Modern fiction reviews", "Creative writing", "Poetry analysis"],
    "Visual Arts": ["Oil painting tutorial", "Digital illustration", "Art history", "Graphic design trends"],
    "Philosophy": ["Stoicism", "Existentialism", "Ethics and Morality", "Eastern philosophy"],
    "History": ["World War II history", "Ancient Civilizations", "History of the Silk Road", "Industrial Revolution"],
    "Psychology": ["Cognitive biases", "Behavioral psychology", "Mental health awareness", "Child development"],
    "Sociology": ["Urban planning", "Social movements", "Demographic shifts", "Cultural anthropology"],
    "Politics": ["Election analysis", "Geopolitics", "Public policy", "Political theory"],
    "Automotive": ["Electric vehicle tech", "Classic car restoration", "F1 technical analysis", "Off-roading"],
    "Aviation": ["Commercial pilot life", "Air crash investigation", "Future of flight", "Private jets"],
    "Photography": ["Portrait lighting", "Landscape photography", "Camera gear reviews", "Film photography"],
    "Sustainability": ["Renewable energy", "Zero waste living", "Circular economy", "Ocean conservation"],
    "Education/Pedagogy": ["Teaching methods", "EdTech trends", "Montessori education", "Learning science"],
    "Podcasts & Talk": ["Long-form interviews", "Roundtable discussions", "Joe Rogan style", "TED Talks"],
    "True Crime": ["Unsolved mysteries", "Famous trials", "Cold case files"],
    "Documentary": ["National Geographic", "Nature documentaries", "Human interest stories", "Investigative journalism"],
    "Parenting": ["Early childhood development", "Modern parenting tips", "Family vlogs", "Educational toys"],
    "Self-Improvement": ["Habit building", "Public speaking", "Time management", "Emotional intelligence"],
    "Woodworking": ["Furniture building", "Carpentry tips", "Epoxy resin art", "Hand tool skills"],
    "Gardening": ["Urban farming", "Hydroponics", "Permaculture", "Houseplant care"],
    "Collectibles": ["TCG collecting", "Vintage toys", "Sneaker culture", "Antiques Roadshow"],
    "Outdoor Sports": ["Rock climbing", "Surfing", "Hiking trails", "Survival skills"],
    "DIY/Crafts": ["Pottery", "Knitting/Crochet", "Glass blowing", "Leatherworking"],
    "Global News": ["BBC World Service", "Al Jazeera", "Reuters reports", "International crisis"],
    "Local News": ["Community issues", "Regional politics", "Local events", "Hyper-local reporting"],
    "Investigative": ["Undercover reporting", "Documentary exposes", "Whistleblower stories", "Financial fraud"],
    "ASMR": ["Relaxation sounds", "Ambience 4k", "Study music", "Focus sounds"],
    "Pets & Animals": ["Dog training", "Exotic pets", "Veterinary science", "Animal behavior"],
    "Spirituality": ["Meditation", "Comparative religion", "Modern spirituality", "Mindfulness"],
    "Law": ["Legal breakdowns", "Supreme court analysis", "Contract law", "Criminal defense"],
    "Language Learning": ["Polyglot tips", "Linguistics", "Language history", "ESL lessons"],
    "Comedy": ["Stand-up specials", "Satire", "Sketch comedy", "Humor analysis"],
    "Architecture": ["Modernist architecture", "Skyscraper design", "Urban design", "Historical landmarks"],
    "Astronomy": ["Telescope reviews", "Stargazing tips", "Solar system exploration", "NASA updates"],
    "Mythology": ["Greek myths", "Norse mythology", "Folklore", "Legend analysis"],
    "Career Advice": ["Resume building", "Interview prep", "Corporate culture", "Remote work tips"],
    "Military Tech": ["Modern weaponry", "Military strategy", "Defense industry", "Historical battles"]
}

CAT_COLORS = ["blue", "red", "bright_blue", "orange3", "purple", "deep_pink", "hot_pink", "chartreuse4"]
def get_cat_style(category):
    idx = sum(ord(c) for c in category) % len(CAT_COLORS)
    return CAT_COLORS[idx]

def is_english(title, description=""):
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        text = f"{title} {description}".strip()
        if not text or len(text) < 10:
            return False
        return detect(text) == 'en'
    except Exception:
        return True

class YDLSilentLogger:
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

def get_ydl_opts(verify_subtitles=True, cookies_path=None):
    opts = {
        'logger': YDLSilentLogger(),
        'quiet': True,
        'extract_flat': False,
        'search_filter': 'relevance', 
        'ignoreerrors': False,
        'no_warnings': True,
        'writesubtitles': verify_subtitles,
        'writeautomaticsub': verify_subtitles,
        'skip_download': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    if cookies_path and Path(cookies_path).exists():
        opts['cookiefile'] = str(cookies_path)
    return opts

def count_existing_categories():
    counts = {cat: 0 for cat in CATEGORIES.keys()}
    if not Path(VIDEO_FILE).exists():
        return counts
    current_cat = None
    with open(VIDEO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Category:"):
                current_cat = line.split(":", 1)[1].strip()
            elif line and not line.startswith("#"):
                if current_cat in counts:
                    counts[current_cat] += 1
                else:
                    counts[current_cat] = counts.get(current_cat, 0) + 1
    return counts

def discover_pro_videos(target_count=50, max_per_channel=20, min_comments=10, min_length_mins=1.0, cookies=None):
    video_links = []
    seen_ids = set()
    channel_counts = {}
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
    console.print(f"[bold blue]Discovery Goal: {target_count} total videos.[/bold blue]")
    console.print(f"[dim]Current: {total_existing} | Needs: {remaining} new links[/dim]\n")
    
    last_saved_category = None
    eta_col = DiscoveryETAColumn(time_window=60)

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}", markup=True),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        eta_col,
        console=console,
        refresh_per_second=4,
        expand=True
    ) as progress:
        task = progress.add_task("Total Progress...", total=target_count, completed=total_existing)
        query_pool = []
        for cat, kw_list in CATEGORIES.items():
            for kw in kw_list:
                query_pool.append({"cat": cat, "kw": kw})
        random.shuffle(query_pool)

        try:
            import yt_dlp
        except ImportError:
            progress.console.print("[bold red][ERR] Dependency 'yt-dlp' is not installed.[/bold red]")
            return []

        with yt_dlp.YoutubeDL(get_ydl_opts(verify_subtitles=False, cookies_path=cookies)) as ydl:
            try:
                while (total_existing + len(video_links)) < target_count:
                    current_balance = {cat: existing_stats.get(cat, 0) for cat in CATEGORIES}
                    for v in video_links:
                        current_balance[v['category']] += 1
                    
                    min_count = min(current_balance.values())
                    under_represented = [cat for cat, count in current_balance.items() if count == min_count]
                    target_cat = random.choice(under_represented)
                    
                    cat_kws = [q for q in query_pool if q['cat'] == target_cat]
                    query_obj = random.choice(cat_kws)
                    base_query = query_obj['kw']
                    semantic_seeds = ["lecture", "explained", "2025", "tutorial", "full course", "session", "talk", "guide"]
                    query = f"{base_query} {random.choice(semantic_seeds)}"
                    
                    display_cat = (target_cat[:12] + '..') if len(target_cat) > 14 else target_cat
                    cat_style = get_cat_style(target_cat)
                    rej_summary = f"[dim]X:{sum(rejections.values())}(S:{rejections['Shorts']}|C:{rejections['Low Comments']})[/dim]"
                    progress.update(task, description=f"[{cat_style}]{display_cat:14}[/{cat_style}] {rej_summary}")

                    search_success = False
                    for attempt in range(3):
                        try:
                            results = ydl.extract_info(f"ytsearch40:{query}", download=False)
                            search_success = True
                            break 
                        except Exception as e:
                            err_str = str(e).lower()
                            if "429" in err_str or "too many requests" in err_str:
                                wait_sec = 30 * (attempt + 1)
                                time.sleep(wait_sec)
                                continue 
                            if "rate-limited" in err_str or "try again later" in err_str or "sign in" in err_str:
                                time.sleep(20 * (attempt + 1))
                                break 
                            break 

                    if not search_success or not results or 'entries' not in results:
                        continue

                    time.sleep(random.uniform(2.0, 4.0))

                    for entry in results['entries']:
                        if not entry: continue
                        if (total_existing + len(video_links)) >= target_count: break
                        v_id = entry.get('id')
                        if not v_id or v_id in seen_ids:
                            if v_id: rejections["Duplicate"] += 1
                            continue
                        
                        availability = entry.get('availability')
                        if (availability and availability != 'public') or entry.get('is_private'):
                            rejections["Blocked/Private"] += 1
                            continue
                        if entry.get('live_status') in ['is_live', 'is_upcoming']:
                            rejections["Blocked/Private"] += 1
                            continue

                        uploader = entry.get('uploader_id') or entry.get('uploader') or "Unknown"
                        if channel_counts.get(uploader, 0) >= max_per_channel:
                            rejections["Other"] += 1 
                            continue
                        
                        title = entry.get('title', '')
                        if not is_english(title): 
                            rejections["Other"] += 1 
                            continue
                        
                        duration = entry.get('duration', 0) or 0
                        view_count = entry.get('view_count', 0) or 0
                        comment_count = entry.get('comment_count')
                        
                        if duration < (min_length_mins * 60):
                            rejections["Shorts"] += 1
                            continue
                        if comment_count is None:
                            if min_comments > 0: 
                                rejections["Low Comments"] += 1
                                continue
                        elif comment_count < min_comments:
                            rejections["Low Comments"] += 1
                            continue

                        video_obj = {
                            'id': v_id, 'title': title, 'url': f"https://www.youtube.com/watch?v={v_id}",
                            'category': target_cat, 'views': view_count,
                            'duration': f"{duration//60}:{duration%60:02d}" if duration else "N/A"
                        }
                        video_links.append(video_obj)
                        seen_ids.add(v_id)
                        channel_counts[uploader] = channel_counts.get(uploader, 0) + 1
                        
                        with open(VIDEO_FILE, "a", encoding="utf-8") as f:
                            if target_cat != last_saved_category:
                                f.write(f"\n# Category: {target_cat}\n")
                                last_saved_category = target_cat
                            f.write(f"{video_obj['url']}\n")

                        progress.advance(task)
                        eta_col.update_history(progress.tasks[task].completed)
                        time.sleep(0.01)
            except KeyboardInterrupt:
                progress.console.print("\n[bold red][HALT] Discovery interrupted by user.[/bold red]")
                pass

    return video_links

def main():
    parser = argparse.ArgumentParser(description="Goal-Aware Video Discovery")
    parser.add_argument("--count", type=parse_count, default=10)
    parser.add_argument("--max-per-channel", type=int, default=20)
    parser.add_argument("--min-comments", type=int, default=10)
    parser.add_argument("--min-length", type=float, default=1.0)
    parser.add_argument("--cookies", type=str, default=None)
    args = parser.parse_args()

    try:
        found = discover_pro_videos(args.count, args.max_per_channel, args.min_comments, args.min_length, args.cookies)
        if not found:
            console.print("[yellow]No new videos found in this session.[/yellow]")
            return
        console.print("\n")
        table = Table(title=f"Discovered {len(found)} High-Quality Videos", title_style="bold cyan")
        table.add_column("Category", style="magenta", no_wrap=True)
        table.add_column("Title", style="white", overflow="ellipsis", max_width=50)
        table.add_column("Views", justify="right", style="green")
        table.add_column("Duration", justify="right", style="cyan")
        for v in found:
            table.add_row(v['category'], v['title'], f"{v['views']:,}", v['duration'])
        console.print(table)
        console.print(f"\n[bold green]Session complete. {len(found)} links were checkpointed to {VIDEO_FILE}![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red][ERR] Discovery failed: {e}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red][HALT] Discovery interrupted by user.[/bold red]")
        sys.exit(0)
