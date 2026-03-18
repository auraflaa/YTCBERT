import os
import sys
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'video_pipeline'))

from utils.helpers import get_video_stats_batch
from visualize_diversity import parse_video_list

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

if not api_key:
    # Just fetch a hardcoded API key if it exists in the environment or user's project
    print("No API key found in .env")
    sys.exit(1)

# Grab 3 videos from video.txt
videos = parse_video_list('video_pipeline/video.txt')
if not videos:
    print("No videos in video.txt")
    sys.exit(1)

first_3 = []
import re
for v in videos[:3]:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", v["url"])
    if match:
        first_3.append(match.group(1))

print(f"Testing IDs: {first_3}")
stats = get_video_stats_batch(first_3, api_key)
print(json.dumps(stats, indent=2))
