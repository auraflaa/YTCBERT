from youtube_transcript_api import YouTubeTranscriptApi
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import sys

def test_extraction(video_id="xOS0BhhdUbo"):
    print(f"=== Testing YouTube Scrapers for Video ID: {video_id} ===")
    
    # 1. Test Transcript
    print("\n[1] Testing Transcript Extraction (youtube-transcript-api)...")
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        text = "\n".join(s.text for s in transcript)
        print("✅ SUCCESS! Retrieved transcript:")
        print(f"    Preview: \"{text[:100]}...\"")
    except Exception as e:
        print(f"❌ FAILED! Transcript blocked or missing: {e}")
        print("    -> Conclusion: Your IP is still blocked by YouTube for transcripts.")

    # 2. Test Comments
    print("\n[2] Testing Comment Extraction (youtube-comment-downloader)...")
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        downloader = YoutubeCommentDownloader()
        
        comments_found = 0
        for item in downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR):
            comments_found += 1
            if comments_found == 1:
                print("✅ SUCCESS! Retrieved comments:")
                print(f"    First Comment Preview: \"{item.get('text', '')[:100]}...\"")
            if comments_found >= 5:
                break
        
        if comments_found == 0:
            print("❌ FAILED! Could not find any comments.")
            
    except Exception as e:
        print(f"❌ FAILED! Comment scraping blocked or missing: {e}")

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "xOS0BhhdUbo"
    test_extraction(vid)
