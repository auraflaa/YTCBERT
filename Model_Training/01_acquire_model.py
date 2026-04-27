"""
Model Training/fetch_model.py
----------------------------
Downloads Google's FLAN-T5 Base model and tokenizer locally 
to the ./flan-t5-base directory for offline training.
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
import os

MODEL_ID = "google/flan-t5-base"
SAVE_DIR = Path(__file__).resolve().parent / "flan-t5-base"

def fetch():
    print(f"Initializing download of {MODEL_ID}...")
    
    # 1. Download/Load Tokenizer
    print("Fetching tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    
    # 2. Download/Load Model
    print("Fetching model weights (this may take a few minutes)...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    
    # 3. Save locally
    print(f"Saving to {SAVE_DIR}...")
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    
    print("\n✅ Model and tokenizer saved successfully!")
    print(f"Location: {SAVE_DIR}")

if __name__ == "__main__":
    fetch()
