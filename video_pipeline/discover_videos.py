"""
Broad Video Discovery for diverse YouTube datasets.

Step 0 of the YTCBERT pipeline.
This script performs automated video discovery based on 12 categorized niches.
Features:
- Goal-Aware Discovery: Treats --count as a total target for video.txt.
- Auto-Balancing: Prioritizes searching for least-represented categories.
- English-Only Filtering: Uses keyword entropy for quality control.
- Real-time Checkpointing: Saves links as they are found.
- Resilient Halts: Graceful Ctrl+C handling with session summaries.
- Multi-Tier Rate Limiting: Handles both transient and hard blocks.
"""

import argparse
import random
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, ProgressColumn
from collections import deque

from utils.helpers import extract_video_id, parse_count

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
        # Keep only the last 'time_window' seconds of data
        while self.history and self.history[0][0] < (now - self.time_window):
            self.history.popleft()

    def render(self, task):
        if not task.total or len(self.history) < 2:
            return Text("-:--:--", style="dim")
        
        # Thruput = (Recent Δ Items) / (Recent Δ Time)
        first_time, first_count = self.history[0]
        last_time, last_count = self.history[-1]
        
        # Calculate throughput over the time window
        delta_time = last_time - first_time
        delta_items = last_count - first_count
        
        # If we just started or no items in window, fallback to long-term session average
        if delta_time < 5 or delta_items <= 0:
            if task.elapsed > 0 and task.completed > 0:
                items_per_sec = task.completed / task.elapsed
            else:
                return Text("-:--:--", style="dim")
        else:
            items_per_sec = delta_items / delta_time

        remaining_count = task.total - task.completed
        remaining_seconds = (remaining_count / items_per_sec) if items_per_sec > 0 else 0
        
        # Calculated projected completion
        total_estimated_seconds = task.elapsed + remaining_seconds

        
        # Formatting: Use distinct labels and formats
        # Duration: e.g. "45m 20s" or "1h 12m"
        total_m, total_s = divmod(int(total_estimated_seconds), 60)
        total_h, total_m = divmod(total_m, 60)
        
        if total_h > 0:
            total_str = f"{total_h}h {total_m}m"
        else:
            total_str = f"{total_m}m {total_s}s"
        
        # Absolute Finish Time (Clock)
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
BASE_DIR = Path(__file__).parent
VIDEO_FILE = BASE_DIR / "video.txt"
console = Console()

# Dynamic Color Map for Categories (Avoiding yellow/green/cyan/magenta used elsewhere)
CAT_COLORS = ["blue", "red", "bright_blue", "orange3", "purple", "deep_pink", "hot_pink", "chartreuse4"]
def get_cat_style(category):
    """Deterministically maps a category string to a color from CAT_COLORS."""
    idx = sum(ord(c) for c in category) % len(CAT_COLORS)
    return CAT_COLORS[idx]

# Category definition for balanced vertical variety
CATEGORIES = {
    # --- TECH & DATA (The Core) ---
    "AI & Deep Learning": ["Karpathy", "Neural Networks", "LLM architecture", "Transformer models"],
    "Data Science / Python": ["Krish Naik", "Scikit-learn tutorial", "Data visualization", "SQL for Data Science"],
    "Tech Ethics & News": ["AI Alignment", "Privacy laws tech", "Silicon Valley news", "Tech monopoly debate"],
    "Software Engineering": ["System Design", "Microservices architecture", "Clean Code principles", "LeetCode solutions"],
    "Cybersecurity": ["Penetration testing", "Zero trust architecture", "Cyber attack analysis", "Encryption explained"],
    "Cloud Computing": ["AWS vs Azure vs GCP", "Docker and Kubernetes", "Serverless architecture", "Cloud migration"],

    # --- SCIENCE & ENGINEERING ---
    "Physics & Space": ["PBS Space Time", "Astrophysics", "Quantum mechanics", "James Webb Telescope"],
    "Biology & Medicine": ["Molecular biology", "Genetics explained", "Medical breakthroughs", "Neuroscience"],
    "General Engineering": ["Structural engineering", "Mechanical design", "Mark Rober", "Veritasium"],
    "Mathematics": ["Number Theory", "Linear Algebra", "Calculus visual", "3Blue1Brown"],
    "Chemistry": ["Chemical reactions", "Organic chemistry", "Material science", "Periodic table deep dive"],

    # --- BUSINESS & FINANCE ---
    "Personal Finance": ["Investment strategies", "Index funds explained", "Credit score optimization", "Retirement planning"],
    "Macroeconomics": ["Global inflation", "Central banks", "Supply chain crisis", "Economic history"],
    "Cryptocurrency/Web3": ["Blockchain technology", "Ethereum smart contracts", "Bitcoin news", "DeFi explained"],
    "Startups & VC": ["Y Combinator", "Pitch deck analysis", "Venture capital trends", "SaaS business model"],
    "Real Estate": ["Property investment", "Housing market analysis", "Commercial real estate", "House flipping"],

    # --- LIFESTYLE & CULTURE ---
    "Travel & Tourism": ["Luxury travel", "Budget backpacking", "Hidden gems Japan", "Travel documentary"],
    "Fashion & Style": ["Streetwear trends", "Luxury brand history", "Sustainable fashion", "Watch collecting"],
    "Interior Design": ["Minimalist home decor", "Architectural Digest", "Small apartment ideas", "DIY home renovation"],
    "Cooking & Food": ["Gordon Ramsay", "Street food tour", "Molecular gastronomy", "Traditional Italian recipes"],
    "Fitness & Health": ["Strength training", "Longevity science", "Nutrition myths", "Marathon preparation"],

    # --- ARTS & ENTERTAINMENT ---
    "Film & Cinema": ["Video essay movies", "Cinematography analysis", "Film history", "Screenwriting tips"],
    "Music Theory": ["Jazz improvisation", "Music production tutorial", "Synthesizer history", "Musicology"],
    "Gaming Culture": ["Esports industry", "Game design analysis", "Retro gaming", "Speedrunning documentary"],
    "Literature & Books": ["Classic literature", "Modern fiction reviews", "Creative writing", "Poetry analysis"],
    "Visual Arts": ["Oil painting tutorial", "Digital illustration", "Art history", "Graphic design trends"],

    # --- HUMANITIES & SOCIAL SCIENCES ---
    "Philosophy": ["Stoicism", "Existentialism", "Ethics and Morality", "Eastern philosophy"],
    "History": ["World War II history", "Ancient Civilizations", "History of the Silk Road", "Industrial Revolution"],
    "Psychology": ["Cognitive biases", "Behavioral psychology", "Mental health awareness", "Child development"],
    "Sociology": ["Urban planning", "Social movements", "Demographic shifts", "Cultural anthropology"],
    "Politics": ["Election analysis", "Geopolitics", "Public policy", "Political theory"],

    # --- SPECIALIZED NICHES ---
    "Automotive": ["Electric vehicle tech", "Classic car restoration", "F1 technical analysis", "Off-roading"],
    "Aviation": ["Commercial pilot life", "Air crash investigation", "Future of flight", "Private jets"],
    "Photography": ["Portrait lighting", "Landscape photography", "Camera gear reviews", "Film photography"],
    "Sustainability": ["Renewable energy", "Zero waste living", "Circular economy", "Ocean conservation"],
    "Education/Pedagogy": ["Teaching methods", "EdTech trends", "Montessori education", "Learning science"],

    # --- HUMAN EXPERIENCE ---
    "Podcasts & Talk": ["Long-form interviews", "Roundtable discussions", "Joe Rogan style", "TED Talks"],
    "True Crime": ["Unsolved mysteries", "Forensic analysis", "Famous trials", "Cold case files"],
    "Documentary": ["National Geographic", "Nature documentaries", "Human interest stories", "Investigative journalism"],
    "Parenting": ["Early childhood development", "Modern parenting tips", "Family vlogs", "Educational toys"],
    "Self-Improvement": ["Habit building", "Public speaking", "Time management", "Emotional intelligence"],

    # --- CRAFTS & HOBBIES ---
    "Woodworking": ["Furniture building", "Carpentry tips", "Epoxy resin art", "Hand tool skills"],
    "Gardening": ["Urban farming", "Hydroponics", "Permaculture", "Houseplant care"],
    "Collectibles": ["TCG collecting", "Vintage toys", "Sneaker culture", "Antiques Roadshow"],
    "Outdoor Sports": ["Rock climbing", "Surfing", "Hiking trails", "Survival skills"],
    "DIY/Crafts": ["Pottery", "Knitting/Crochet", "Glass blowing", "Leatherworking"],

    # --- NEWS & CURRENT EVENTS ---
    "Global News": ["BBC World Service", "Al Jazeera", "Reuters reports", "International crisis"],
    "Local News": ["Community issues", "Regional politics", "Local events", "Hyper-local reporting"],
    "Investigative": ["Undercover reporting", "Documentary exposes", "Whistleblower stories", "Financial fraud"],

    # --- MISCELLANEOUS ---
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

def is_english(title, description=""):
    """
    Performs a heuristic check for English content using langdetect.
    Checks the combined title and description for higher accuracy.
    """
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        text = f"{title} {description}".strip()
        if not text or len(text) < 10:
            return False
        return detect(text) == 'en'
    except ImportError:
        # Fallback to true if we can't check, rather than crashing
        return True
    except Exception:
        return False

class YDLSilentLogger:
    """Custom logger to keep the console clean from yt-dlp internal noise."""
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        # We handle critical errors (like rate limits) ourselves via the exception message
        pass

def get_ydl_opts(verify_subtitles=True, cookies_path=None):
    """Returns standardized configuration for yt-dlp."""
    opts = {
        'logger': YDLSilentLogger(),
        'quiet': True,
        'extract_flat': False, # Extract full info to check views/duration
        'search_filter': 'relevance', 
        'ignoreerrors': False,
        'no_warnings': True,
        'writesubtitles': verify_subtitles,
        'writeautomaticsub': verify_subtitles,
        'skip_download': True, # We only want metadata
        # Anti-Bot Headers
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    if cookies_path and Path(cookies_path).exists():
        opts['cookiefile'] = str(cookies_path)
    return opts

def count_existing_categories():
    """Returns a dict of {category: count} based on current video.txt."""
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
                    # In case of manual edits or unknown categories
                    counts[current_cat] = counts.get(current_cat, 0) + 1
    return counts

def discover_pro_videos(target_count=50, max_per_channel=20, min_comments=10, min_length_mins=1.0, cookies=None):
    """
    Main discovery logic. Goal-aware and balance-aware.
    It prioritizes categories that are currently under-represented in video.txt.
    """
    video_links = []
    seen_ids = set()
    channel_counts = {} # Track how many videos we've taken from each channel in THIS session
    
    # Rejection Stats for transparency
    rejections = {"Shorts": 0, "Low Comments": 0, "Duplicate": 0, "Blocked/Private": 0, "Other": 0}

    
    # 1. Audit the existing file
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
    
    from rich.progress import MofNCompleteColumn, TimeElapsedColumn, TaskProgressColumn
    last_saved_category = None

    # 2. Setup Discovery UI
    from rich.progress import MofNCompleteColumn, TaskProgressColumn
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
        
        # Build a pool of all possible queries
        query_pool = []
        for cat, kw_list in CATEGORIES.items():
            for kw in kw_list:
                query_pool.append({"cat": cat, "kw": kw})
        random.shuffle(query_pool)

        try:
            import yt_dlp
        except ImportError:
            progress.console.print("[bold red][ERR] Dependency 'yt-dlp' is not installed.[/bold red]")
            progress.console.print("[yellow]Please run: pip install -r requirements.txt[/yellow]")
            return []

        with yt_dlp.YoutubeDL(get_ydl_opts(verify_subtitles=False, cookies_path=cookies)) as ydl:
            try:
                while (total_existing + len(video_links)) < target_count:
                    # BALANCE LOGIC: Find which category is currently the "thinnest"
                    # (Existing in file + Discovered in this session)
                    current_balance = {cat: existing_stats.get(cat, 0) for cat in CATEGORIES}
                    for v in video_links:
                        current_balance[v['category']] += 1
                    
                    min_count = min(current_balance.values())
                    under_represented = [cat for cat, count in current_balance.items() if count == min_count]
                    target_cat = random.choice(under_represented)
                    
                    # Pick a random keyword from the target category
                    cat_kws = [q for q in query_pool if q['cat'] == target_cat]
                    query_obj = random.choice(cat_kws)
                    base_query = query_obj['kw']
                    
                    # SEMANTIC JITTER: Use high-value terms instead of random letters
                    semantic_seeds = ["lecture", "explained", "2025", "tutorial", "full course", "session", "talk", "guide"]
                    seed = random.choice(semantic_seeds)
                    query = f"{base_query} {seed}"
                    
                    # COMPACT TELEMETRY: Shorten category and rejections to prevent terminal wrapping
                    display_cat = (target_cat[:12] + '..') if len(target_cat) > 14 else target_cat
                    cat_style = get_cat_style(target_cat)
                    rej_summary = f"[dim]X:{sum(rejections.values())}(S:{rejections['Shorts']}|C:{rejections['Low Comments']})[/dim]"
                    progress.update(task, description=f"[{cat_style}]{display_cat:14}[/{cat_style}] {rej_summary}")



                    # Multi-stage retry for Transient RPM limits
                    search_success = False
                    for attempt in range(3):
                        try:
                            # HYPER-DISCOVERY: Reduced to 40 for stability (100 is too aggressive for scrapers)
                            results = ydl.extract_info(f"ytsearch40:{query}", download=False)
                            search_success = True
                            break # Success!


                        except Exception as e:
                            err_str = str(e).lower()
                            # Tier 1: Transient RPM (Too Many Requests / 429)
                            if "429" in err_str or "too many requests" in err_str:
                                wait_sec = 30 * (attempt + 1)
                                progress.update(task, description=f"[yellow]RPM Limit (429)[/yellow] - Waiting {wait_sec}s...")
                                time.sleep(wait_sec)
                                continue # Retry the same query
                            
                            # Tier 2: Transient Bot Block (Sign in to confirm / Try again later)
                            if "rate-limited" in err_str or "try again later" in err_str or "sign in" in err_str:
                                wait_sec = 20 * (attempt + 1)
                                progress.update(task, description=f"[yellow]Block Detected[/yellow] - Backing off {wait_sec}s...")
                                if attempt == 2:
                                    progress.console.print("[bold red][TIP] YouTube is blocking these searches. Try using --cookies <path_to_cookies.txt> to bypass.[/bold red]")
                                time.sleep(wait_sec)
                                break # Give up on this specific niche/seed and try a completely new query

                            
                            # Other errors (e.g. network) - Just skip silently
                            break # Give up on this specific niche/seed

                    if not search_success or not results or 'entries' not in results:
                        continue

                    # Add a jittered delay between successful searches
                    time.sleep(random.uniform(2.0, 4.0))

                    for entry in results['entries']:
                            if not entry: continue
                            if (total_existing + len(video_links)) >= target_count: break
                            
                            v_id = entry.get('id')
                            if not v_id or v_id in seen_ids:
                                if v_id: rejections["Duplicate"] += 1
                                continue
                            
                            # Filter out restricted or non-public videos without extra latency
                            availability = entry.get('availability')
                            if (availability and availability != 'public') or entry.get('is_private'):
                                rejections["Blocked/Private"] += 1
                                continue
                                
                            # Also skip live streams to ensure we get static content
                            if entry.get('live_status') in ['is_live', 'is_upcoming']:
                                rejections["Blocked/Private"] += 1
                                continue

                            uploader = entry.get('uploader_id') or entry.get('uploader') or "Unknown"
                            if channel_counts.get(uploader, 0) >= max_per_channel:
                                rejections["Other"] += 1 # Over channel cap
                                continue
                            
                            title = entry.get('title', '')
                            if not is_english(title): 
                                rejections["Other"] += 1 # Non-English
                                continue
                            
                            duration = entry.get('duration', 0) or 0
                            view_count = entry.get('view_count', 0) or 0
                            comment_count = entry.get('comment_count')
                            
                            if duration < (min_length_mins * 60):
                                rejections["Shorts"] += 1
                                continue
                            
                            # Filter on comments
                            if comment_count is None:
                                if min_comments > 0: 
                                    rejections["Low Comments"] += 1
                                    continue # Skip if hidden/disabled but we demand them
                            elif comment_count < min_comments:
                                rejections["Low Comments"] += 1
                                continue

                            
                            video_obj = {
                                'id': v_id,
                                'title': title,
                                'url': f"https://www.youtube.com/watch?v={v_id}",
                                'category': target_cat,
                                'views': view_count,
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
                            # Update ETA history (fetching Task object from ID)
                            eta_col.update_history(progress.tasks[task].completed)
                            time.sleep(0.01)
            except KeyboardInterrupt:
                progress.console.print("\n[bold red][HALT] Discovery interrupted by user.[/bold red]")
                # We stop searching but still return what we have found so far
                pass

    return video_links

def main():
    """CLI Entry point for goal-aware video discovery."""
    parser = argparse.ArgumentParser(
        description="Goal-Aware Video Discovery: Reaches a total target count while balancing niches."
    )
    parser.add_argument("--count", type=parse_count, default=10, help="Total target count (e.g. 10, 2K, 1M)")
    parser.add_argument("--max-per-channel", type=int, default=20, help="Max videos from a single channel (default: 20)")
    parser.add_argument("--min-comments", type=int, default=10, help="Minimum comments required per video (default: 10)")
    parser.add_argument("--min-length", type=float, default=1.0, help="Minimum video length in minutes (default: 1.0)")
    parser.add_argument("--cookies", type=str, default=None, help="Path to cookies.txt to bypass bot-checks")
    args = parser.parse_args()

    try:
        # Discover and Save in real-time
        found = discover_pro_videos(args.count, args.max_per_channel, args.min_comments, args.min_length, args.cookies)
        
        if not found:
            console.print("[yellow]No new videos found in this session.[/yellow]")
            return

        # Display pretty results table of the session
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
