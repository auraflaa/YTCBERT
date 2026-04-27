import requests
import random
import socket
import time
from pathlib import Path
from utils.helpers import resolve_data_path

# Globally prevent socket hangs from dead free proxies
socket.setdefaulttimeout(15.0)

class ProxyRotator:
    def __init__(self):
        self.proxies = []
        self.proxies_file = resolve_data_path("proxies.txt")
        self._last_refresh = 0
        
    def refresh_pool(self):
        """Loads proxies from a local proxies.txt file if it exists."""
        if self.proxies_file.exists():
            try:
                content = self.proxies_file.read_text(encoding="utf-8")
                pool = [L.strip() for L in content.splitlines() if ":" in L]
                self.proxies = list(set(pool))
            except Exception as e:
                print(f"Error loading {self.proxies_file.name}: {e}")
        else:
            self.proxies = []
        
    def get_proxy(self):
        """Returns a formatted proxy dict for requests.Session or None if using Home IP."""
        # Refresh from file every 60 seconds if file exists
        if time.time() - self._last_refresh > 60:
            self.refresh_pool()
            self._last_refresh = time.time()

        if not self.proxies:
            return None
            
        p = random.choice(self.proxies)
        # Handle auth proxies (user:pass@ip:port)
        if "@" in p:
            return {"http": p, "https": p}
            
        return {
            "http": f"http://{p}",
            "https": f"http://{p}"
        }

rotator = ProxyRotator()
