import sys
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

def test_exception(video_id="s3KnSb9b4Pk"):
    print(f"Testing exception for: {video_id}")
    try:
        YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        print("Success! It has a transcript!?")
    except Exception as e:
        print(f"Exception Type: {type(e).__name__}")
        print(f"Is NoTranscriptFound? {isinstance(e, NoTranscriptFound)}")
        print(f"Is TranscriptsDisabled? {isinstance(e, TranscriptsDisabled)}")
        print(f"Is VideoUnavailable? {isinstance(e, VideoUnavailable)}")
        print(f"Is CouldNotRetrieveTranscript? {isinstance(e, CouldNotRetrieveTranscript)}")
        print(f"Class MRO: {[c.__name__ for c in type(e).__mro__]}")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "s3KnSb9b4Pk"
    test_exception(vid)
