import http.cookiejar
from pathlib import Path
import requests
from rich.console import Console

console = Console()

class YouTubeAuth:
    """
    Central authentication manager for the YouTube pipeline.
    Loads Netscape-format cookies and provides they can be used across 
    requests, yt_dlp, and other scrapers.
    """
    
    def __init__(self, cookie_str: str | None = None):
        self.cookie_str = cookie_str
        self.jar = None
        if cookie_str:
            self.load_cookies(cookie_str)

    def load_cookies(self, cookie_str: str):
        """Loads cookies from a file path or identifies a browser name."""
        self.cookie_str = cookie_str
        
        # Resolve path using our helper (Check data/ first)
        from utils.helpers import resolve_data_path
        target = resolve_data_path(cookie_str)
        
        if target and target.exists():
            try:
                self.jar = http.cookiejar.MozillaCookieJar(str(target.absolute()))
                self.jar.load(ignore_discard=True, ignore_expires=True)
                console.print(f"  [green]✔ Loaded cookies from {target.name}[/green]")
            except Exception as e:
                console.print(f"  [red]✖ Failed to load cookie file: {e}[/red]")
                self.jar = None
        else:
            # Fallback for browser names (handled later in yt-dlp specific logic)
            self.jar = None

    def get_session(self) -> requests.Session:
        """Returns a requests.Session pre-loaded with the cookies."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        if self.jar:
            session.cookies = self.jar
        return session

    def apply_to_ytdlp(self, ydl_opts: dict):
        """Injects authentication into yt-dlp options."""
        if not self.cookie_str:
            return
            
        from utils.helpers import resolve_data_path
        target = resolve_data_path(self.cookie_str)
        
        if target and target.exists():
            ydl_opts['cookiefile'] = str(target.absolute())
        elif self.cookie_str.lower() in ['chrome', 'firefox', 'edge', 'safari', 'opera', 'chromium']:
            ydl_opts['cookiesfrombrowser'] = (self.cookie_str.lower(),)
