"""
Model Training/07_interactive_inference.py
-----------------------------------------
An interactive script to manually test YTCBERT with custom prompts.
"""

import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

# --- CONFIG ---
MT_DIR = Path(__file__).resolve().parent
FINAL_MODEL = MT_DIR / "YTCBERT"

def get_hw_config():
    """Detects available hardware with fallback logic."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            _ = torch.zeros(1, device="xpu")
            return "xpu"
        except: pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def interactive_inference():
    if not FINAL_MODEL.exists():
        print(f"Error: Model not found at {FINAL_MODEL}.")
        return

    device_type = get_hw_config()
    device = torch.device(device_type)
    print(f">>> Initializing on: {device_type.upper()}")

    # Load Model & Tokenizer
    print(f"Loading YTCBERT...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(FINAL_MODEL),
        torch_dtype=torch.float32 # Use float32 for maximum compatibility on CPU/GPU
    ).to(device)
    model.eval()

    print("\n" + "="*30)
    print(" YTCBERT INTERACTIVE SUMMARY ")
    print(" (Type 'exit' to quit) ")
    print("="*30 + "\n")

    while True:
        try:
            user_input = input("Enter audience comments to summarize: \n> ")
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break

            if not user_input.strip():
                continue

            # Preprocess
            input_text = f"summarize: {user_input.strip()}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

            # Generate
            start_time = time.time()
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )
            latency = time.time() - start_time
            
            summary = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

            print("\n" + "-"*20)
            print(f"AI SUMMARY ({latency:.2f}s):")
            print(summary)
            print("-"*20 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    interactive_inference()
