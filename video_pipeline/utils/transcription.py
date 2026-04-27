import threading
from pathlib import Path
from rich.console import Console
from utils.auth import YouTubeAuth
import re

console = Console()

class RobustTranscriber:
    """
    A Hybrid transcription engine for YouTube!
    Tier 1: Speed (yt-dlp Metadata) - Grabs transcripts in 1 second.
    Tier 2: Reliability (Whisper AI) - Transcribes audio locally if Tier 1 fails.
    """
    
    _model = None  # Singleton for Whisper model to save memory
    _model_lock = threading.Lock()
    
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.auth = YouTubeAuth()

    def set_cookies(self, cookie_str: str | None):
        """Sets the cookie source (browser name or file path) and initializes auth."""
        self.auth.load_cookies(cookie_str) if cookie_str else None

    def get_transcript(self, video_id: str, url: str) -> str:
        # TIER 1: SPEED (YouTube Metadata via yt-dlp)
        try:
            return self._fetch_ytdlp_subtitles(video_id, url)
        except Exception as e:
            # If the video is literally unavailable, don't waste time on AI
            if "unavailable" in str(e).lower():
                raise e
            console.print(f"  [yellow]⚠ YouTube Scrape failed for {video_id}. Switching to Local AI (Whisper)...[/yellow]")

        # TIER 2: RELIABILITY (Local AI Whisper)
        return self._fetch_whisper_ai(video_id, url)

    def _fetch_ytdlp_subtitles(self, video_id: str, url: str) -> str:
        """Fetches the pre-generated YouTube transcript using yt-dlp (Fast)."""
        import yt_dlp
        ydl_opts = {
            'skip_download': True,
            'writeautomaticsub': True,
            'writesubtitles': True,
            'subtitleslangs': ['en'],
            'quiet': True,
            'no_warnings': True,
            'geo_bypass': True,
        }
        
        # Inject Cookies into yt-dlp
        self.auth.apply_to_ytdlp(ydl_opts)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            sub_url = None
            
            # Check manual subs then auto-subs
            en_subs = info.get('subtitles', {}).get('en') or info.get('automatic_captions', {}).get('en')
            if en_subs:
                sub_url = en_subs[0]['url']
                
        if not sub_url:
            raise Exception("No English subtitles found in YouTube metadata")

        # Use an authenticated requests session for the actual content download
        session = self.auth.get_session()
        resp = session.get(sub_url, timeout=10)
        if resp.status_code == 200:
            return self._clean_vtt(resp.text)
        
        raise Exception(f"Failed to download YT Transcript: HTTP {resp.status_code}")

    def _fetch_whisper_ai(self, video_id: str, url: str) -> str:
        """Transcribes the video audio using Local AI (Slow but Unblockable)."""
        import yt_dlp
        video_dir = self.output_root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        audio_file = video_dir / f"{video_id}_audio.m4a"
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio',
            'outtmpl': str(audio_file),
            'quiet': True,
            'no_warnings': True,
            'geo_bypass': True,
            'nocheckcertificate': True,
        }
        
        self.auth.apply_to_ytdlp(ydl_opts)
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            model = self._get_whisper_model()
            # beam_size=2 is a good balance between speed and accuracy
            segments, _ = model.transcribe(str(audio_file), beam_size=2)
            
            return " ".join([seg.text.strip() for seg in segments]).strip()
        finally:
            if audio_file.exists():
                try: audio_file.unlink()
                except: pass

    def _get_whisper_model(self):
        with self._model_lock:
            if RobustTranscriber._model is None:
                from faster_whisper import WhisperModel
                console.print(f"  [cyan]⚡ Initializing Faster-Whisper (base) on CPU (int8)...[/cyan]")
                RobustTranscriber._model = WhisperModel("base", device="cpu", compute_type="int8")
            return RobustTranscriber._model

    def _clean_vtt(self, vtt_text: str) -> str:
        lines = vtt_text.splitlines()
        clean = []
        for line in lines:
            if '-->' in line or not line.strip() or line.strip().upper() == "WEBVTT":
                continue
            line = re.sub(r'<[^>]+>', '', line)
            clean.append(line.strip())
        
        deduped = []
        for c in clean:
            if not deduped or deduped[-1] != c:
                deduped.append(c)
        return " ".join(deduped).strip()
