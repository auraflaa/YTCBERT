"""
Data_Preprocessing/summarizer.py
---------------------------------
Enterprise-grade Batch Summarizer for YouTube Data.
Supports Gemini & Gemma models for benchmarking ground-truth.
"""

import os
import json
import time
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TaskProgressColumn

# --- ROOT AWARENESS & CONFIG ---
CURR_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURR_DIR.parent
load_dotenv(ROOT_DIR / ".env")
console = Console()

# Standardized Paths
SOURCE_DIR = CURR_DIR / "cleaned_output"
SUMMARY_DIR = CURR_DIR / "summaries"

# Model Mapping (Updated for current API)
MODELS = {
    "gemini": "gemma-4-31b-it", # Standard alias
    "gemma":  "gemma-4-31b-it", # Standard alias
    "gemma4": "gemma-4-31b-it",
    "gemma3": "gemma-3-27b-it" 
}

# Rate Limiting Logic (30 RPM for Gemma 4, 15 RPM for Gemma 3)
class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self.last_call = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()

# Global State
STOP_EVENT = threading.Event()
LIMITERS = {
    MODELS["gemma4"]: RateLimiter(30),
    MODELS["gemma3"]: RateLimiter(15)
}

# =============================================================================
# Core Logic
# =============================================================================

def get_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in root .env")
    return genai.Client(api_key=api_key)

def load_video_batch(video_dirs, max_comments=200):
    batch_data = []
    for v_dir in video_dirs:
        c_path = v_dir / "cleaned_comments.json"
        
        if c_path.exists():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    comments = json.load(f).get("comments", [])
                # Higher sample rate for better grounding
                batch_data.append({"video_id": v_dir.name, "comments": [c["text"] for c in comments[:max_comments]]})
            except Exception as e:
                console.print(f"  [red]Error reading {v_dir.name}: {e}[/red]")
    return batch_data

def build_prompt(batch_data):
    # Load explicit prompt from file if available
    p_path = CURR_DIR / "prompt.txt"
    if p_path.exists():
        base_prompt = p_path.read_text(encoding="utf-8")
    else:
        # Fallback to a basic version if prompt.txt is missing
        base_prompt = "You are an expert analyst of YouTube audience discussions. Summarize the comments provided."

    input_json = json.dumps({"videos": batch_data}, indent=2)
    
    return f"""{base_prompt}

INPUT DATA (YouTube comments from viewers):
These are YouTube comments from viewers. Summarize the audience discussion, not the video content.

{input_json}
"""

def repair_json(text):
    """Attempt to fix common LLM JSON errors like unescaped quotes and trailing commas."""
    # 1. Strip non-printable/corrupt characters
    text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
    
    # 2. Fix curly/smart quotes (often echoed back by LLM)
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    
    # 3. Remove trailing commas in arrays/objects
    text = re.sub(r",\s*([\]\}])", r"\1", text)
    
    # 4. Fix unescaped quotes inside strings
    # This regex finds "key": "value with "quotes" inside"
    text = re.sub(r'(?<!\\)"(?!:|,|\s*\}|\s*\])', r'\"', text)
    
    return text

def extract_json(text):
    """Robustly extract JSON from potential Markdown wrappers."""
    text = text.strip()
    
    # Try to find the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1:
        extracted = text[start : end + 1]
    else:
        return text

    # Handle Markdown code blocks specifically if they exist
    if "```" in extracted:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            extracted = match.group(1)
            
    return extracted

def summarize_batch(client, batch_data, model_id, fallback_model_id=None, is_gemma=False, attempts=3):
    prompt = build_prompt(batch_data)
    
    # Standard Gemini supports JSON mode; Gemma (via some APIs) does not
    config_args = {
        'temperature': 0.1 # Lower temperature for more consistent/logical summaries
    }
    if not is_gemma:
        config_args['response_mime_type'] = 'application/json'
        config_args['system_instruction'] = "You are a professional YouTube analyst. Always output valid JSON."
    
    config = types.GenerateContentConfig(**config_args)
    system_p = "You are a professional YouTube analyst. Always output valid JSON."
    
    # Get relevant limiter
    limiter = LIMITERS.get(model_id)

    # STAGE 1: Attempt Primary Model
    for attempt in range(1, attempts + 1):
        try:
            # Enforce RPM limits before API call
            if limiter:
                limiter.wait()

            # Format Hardening: Add a strict instruction on retries for JSON errors
            current_prompt = prompt
            if attempt > 1:
                current_prompt += "\n\nIMPORTANT: Your previous response was invalid JSON. Ensure all double quotes inside your 'summary' and 'key_points' strings are escaped with a backslash (e.g., \\\"quote\\\")."

            if is_gemma:
                full_msg = f"{system_p}\n\n{current_prompt}"
                response = client.models.generate_content(model=model_id, contents=full_msg, config=config)
            else:
                response = client.models.generate_content(model=model_id, contents=current_prompt, config=config)
            
            clean_text = extract_json(response.text)
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                # Attempt one-time repair before failing this attempt
                repaired = repair_json(clean_text)
                return json.loads(repaired)

        except Exception as e:
            err_str = str(e).lower()
            if "json" in err_str:
                console.print(f"  [yellow]JSON Parse Error on {model_id}. Retrying with hardening... (Attempt {attempt}/{attempts})[/yellow]")
                continue
            
            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                if attempt == attempts and fallback_model_id:
                    console.print(f"  [bold yellow]⚠️ Primary model {model_id} exhausted. Switching to backup: {fallback_model_id}[/bold yellow]")
                    return summarize_batch(client, batch_data, fallback_model_id, fallback_model_id=None, is_gemma=True, attempts=1)
                
                wait_time = 30 * attempt
                console.print(f"  [yellow]Rate limited (429) on {model_id}. Retrying in {wait_time}s... (Attempt {attempt}/{attempts})[/yellow]")
                time.sleep(wait_time)
                continue
            
            console.print(f"  [red]LLM Error [{model_id}]: {e}[/red]")
            return None
    return None

def process_all(model_alias, batch_size, force, limit=None, workers=3, max_comments=200):
    SUMMARY_DIR.mkdir(exist_ok=True)
    client = get_client()
    
    model_id = MODELS.get(model_alias.lower(), model_alias)
    is_gemma = "gemma" in model_id.lower()
    
    # Determine fallback (Gemma 3 as backup for Gemma 4)
    fallback_id = MODELS["gemma3"] if model_id == MODELS["gemma4"] else None

    # Find pending work
    all_dirs = [d for d in sorted(SOURCE_DIR.iterdir()) if d.is_dir()]
    to_process = []
    skipped = 0
    for d in all_dirs:
        if force or not (SUMMARY_DIR / f"{d.name}.json").exists():
            to_process.append(d)
        else:
            skipped += 1

    if not to_process:
        console.print("[bold green]✅ All videos already summarized. Use --force to re-generate.[/bold green]")
        return

    if limit:
        to_process = to_process[:limit]

    # Dashboard Header
    console.print(f"\n[bold blue]>>> YTCBERT Summarizer[/bold blue]")
    console.print(f"========================================")
    console.print(f"  [cyan]Model:[/cyan]         {model_id}")
    if fallback_id:
        console.print(f"  [cyan]Backup:[/cyan]        {fallback_id}")
    console.print(f"  [cyan]Workers:[/cyan]       {workers}")
    console.print(f"  [cyan]Max Comments:[/cyan]  {max_comments}")
    console.print(f"  [cyan]Batch Size:[/cyan]    {batch_size}")
    console.print(f"  [cyan]Total Found:[/cyan]   {len(all_dirs)}")
    console.print(f"  [green]Skipped:[/green]       {skipped}")
    console.print(f"  [yellow]To Process:[/yellow]    {len(to_process)}")
    console.print(f"========================================\n")

    # Shared results for failure logging
    failures = []
    lock = threading.Lock()

    def process_batch(i, progress, task):
        if STOP_EVENT.is_set():
            return

        current_dirs = to_process[i : i + batch_size]
        v_ids = [d.name for d in current_dirs]
        
        batch_data = load_video_batch(current_dirs, max_comments=max_comments)
        
        if not batch_data or STOP_EVENT.is_set():
            progress.advance(task)
            return

        try:
            output = summarize_batch(client, batch_data, model_id, fallback_model_id=fallback_id, is_gemma=is_gemma)
            
            if STOP_EVENT.is_set(): return

            if output and "results" in output:
                for res in output["results"]:
                    vid = res.get("video_id")
                    if vid:
                        with open(SUMMARY_DIR / f"{vid}.json", "w", encoding="utf-8") as f:
                            json.dump(res, f, indent=2, ensure_ascii=False)
                # console.print(f"  [green]✓ Batch starting with {v_ids[0]} completed.[/green]")
            else:
                with lock:
                    failures.append((v_ids[0], "No results returned by LLM"))
                console.print(f"  [yellow]WARN: Batch starting with {v_ids[0]} failed (No results).[/yellow]")
        except Exception as e:
            err_msg = str(e)
            with lock:
                failures.append((v_ids[0], err_msg))
            console.print(f"  [red]Thread Error in batch {v_ids[0]}: {err_msg}[/red]")
        
        progress.advance(task)

    try:
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            total_batches = (len(to_process) + batch_size - 1) // batch_size
            task = progress.add_task("Summarizing batches...", total=total_batches)
            
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batch_indices = range(0, len(to_process), batch_size)
                futures = [executor.submit(process_batch, i, progress, task) for i in batch_indices]
                
                # Windows-friendly interruptible wait
                while any(not f.done() for f in futures):
                    if STOP_EVENT.is_set():
                        break
                    try:
                        # Short timeout allows KeyboardInterrupt to propagate
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        STOP_EVENT.set()
                        break

                if STOP_EVENT.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        console.print("\n[bold red][HALT] Workflow interrupted by user. Status saved.[/bold red]")

    if failures:
        fail_file = CURR_DIR / "summaries" / "failures.log"
        with open(fail_file, "a") as f:
            for fid, err in failures:
                f.write(f"[{fid}] Error: {err}\n")
        console.print(f"\n[red]Finished with {len(failures)} failures. See {fail_file.name} for details.[/red]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Batch Summarization Tool")
    parser.add_argument("--model", type=str, default="gemini", help="Model alias (gemini, gemma) or specific model ID")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of videos per LLM call")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--max-comments", type=int, default=200, help="Max comments per video to send to LLM")
    parser.add_argument("--force", action="store_true", help="Re-summarize even if output exists")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    args = parser.parse_args()

    try:
        process_all(args.model, args.batch_size, args.force, args.limit, args.workers, args.max_comments)
    except Exception as e:
        console.print(f"[bold red]FATAL ERROR: {e}[/bold red]")
