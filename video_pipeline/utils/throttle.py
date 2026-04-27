import time
import random
import threading
from rich.console import Console

console = Console()

class DynamicController:
    """
    A thread-safe adaptive throttle that mathematically emulates human browsing entropy 
    across Google's backend infrastructure.
    """
    def __init__(self, batch_size_min: int = 40, batch_size_max: int = 70, cooldown_min_sec: int = 180, cooldown_max_sec: int = 300):
        self.lock = threading.Lock()
        
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.cooldown_min_sec = cooldown_min_sec
        self.cooldown_max_sec = cooldown_max_sec
        
        self.req_count = 0
        self.current_batch_target = random.randint(self.batch_size_min, self.batch_size_max)
        
        self.consecutive_429s = 0
        self.base_delay = 2.5 # Minimum baseline human delay
        self.hard_cooldown_until = 0 # Timestamp for long-term blocks
        
    def wait(self):
        """Invoke before making a network request."""
        # Hard Cooldown Check (Non-blocking check outside lock for performance)
        now = time.time()
        if now < self.hard_cooldown_until:
            wait_time = int(self.hard_cooldown_until - now)
            console.print(f"\n[bold red][COOLDOWN][/bold red] Hard IP Block active. Waiting {wait_time}s for reset...")
            time.sleep(wait_time)

        with self.lock:
            self.req_count += 1
            trigger_cooldown = (self.req_count >= self.current_batch_target)
            
            if trigger_cooldown:
                # Reset batch counter for the next session
                self.req_count = 0
                self.current_batch_target = random.randint(self.batch_size_min, self.batch_size_max)
                cooldown = random.randint(self.cooldown_min_sec, self.cooldown_max_sec)
                
            multiplier = 1.0 + (self.consecutive_429s * 0.5)
            delay = random.uniform(self.base_delay, self.base_delay + 2.5) * multiplier

        if trigger_cooldown:
            console.print(f"\n[bold cyan][THROTTLE][/bold cyan] Session entropy boundary reached. Engaging human cooldown for {cooldown}s...")
            time.sleep(cooldown)
            console.print(f"[bold cyan][THROTTLE][/bold cyan] Session resumed.")
        else:
            time.sleep(delay)

    def report_success(self):
        """Invoke upon successful transparent payload extraction."""
        with self.lock:
            if self.consecutive_429s > 0:
                self.consecutive_429s = max(0, self.consecutive_429s - 1)

    def report_429(self, long_block: bool = False):
        """Invoke if YouTube denies the request via Captcha wall or 429."""
        with self.lock:
            self.consecutive_429s += 1
            if long_block:
                # 20-30 minute heavy reset
                wait_time = random.randint(1200, 1800)
                self.hard_cooldown_until = time.time() + wait_time
                console.print(f"  [bold red][THROTTLE][/bold red] Persistent IP block detected! Engaging hard cooldown for {wait_time}s...")
                return

            penalty = 15 * self.consecutive_429s
            console.print(f"  [bold red][THROTTLE][/bold red] Behavioral wall detected! Escalating dynamic backoff multiplier to x{1.0 + (self.consecutive_429s * 0.5)} (Penalty: {penalty}s)")
        
        # Exponential sleep outside the lock to avoid blocking other threads entirely from sleeping
        time.sleep(penalty)

# Global singleton instantiated securely
global_throttle = DynamicController()
