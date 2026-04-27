"""
Model Inference/inference.py
---------------------------
Runs inference and calculates performance metrics (ROUGE, Latency).
Optimized for Intel GPU (XPU), NVIDIA (CUDA), and CPU.
"""

import torch
import time
import evaluate
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

# --- CONFIG ---
# This looks for the model in: ...\YTCBERT\Model_Training\YTCBERT
MT_DIR = Path(__file__).resolve().parent
FINAL_MODEL = MT_DIR / "YTCBERT"
DATASET_FILE = MT_DIR / "summarization_dataset.jsonl"

def get_hw_config():
    """Detects available hardware with fallback logic."""
    # Detect Intel GPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            # Basic sanity check: Try to create a small tensor on XPU
            _ = torch.zeros(1, device="xpu")
            return "xpu"
        except Exception as e:
            print(f">>> XPU detected but failed to initialize: {e}")
            print(">>> Falling back to CPU/CUDA...")
    
    # Detect NVIDIA GPU
    if torch.cuda.is_available():
        return "cuda"
    
    return "cpu"

def run_inference(num_samples=10):
    if not FINAL_MODEL.exists():
        print(f"Error: Model not found at {FINAL_MODEL}.")
        print("Check if the folder name is 'YTCBERT' inside your Model_Training directory.")
        return

    device_type = get_hw_config()
    device = torch.device(device_type)
    print(f">>> Running on: {device_type.upper()}")

    # 1. Load Metric & Data
    print("Loading metrics and dataset...")
    rouge = evaluate.load("rouge")
    raw_dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    
    # ✅ IMPORTANT: Split with the SAME seed used in training to isolate the test set
    split_ds = raw_dataset.train_test_split(test_size=0.1, seed=42)
    test_set = split_ds["test"]
    
    # Select samples from the actual unseen test set
    max_test_samples = min(len(test_set), num_samples)
    samples = test_set.select(range(max_test_samples))

    # 2. Load Model & Tokenizer
    print(f"Loading YTCBERT from {FINAL_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(FINAL_MODEL),
        torch_dtype=torch.bfloat16 if device_type == "xpu" else torch.float32
    ).to(device)
    model.eval()

    predictions = []
    references = []
    latencies = []

    print("\n" + "="*20 + " GENERATING & EVALUATING " + "="*20)

    for i, example in enumerate(samples):
        raw_text = example["input"]
        ground_truth = example["output"]

        # 3. Preprocess
        if raw_text.startswith("Summarize audience comments:"):
            clean_text = raw_text.replace("Summarize audience comments:", "").strip()
        else:
            clean_text = raw_text.strip()
        input_text = f"summarize: {clean_text}"

        inputs = tokenizer(input_text, return_tensors="pt", max_length=384, truncation=True).to(device)

        # 4. Measure Generation Time
        start_time = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_length=96,
                num_beams=4,
                early_stopping=True
            )
        latency = time.time() - start_time
        
        prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Store for metrics
        predictions.append(prediction)
        references.append(ground_truth)
        latencies.append(latency)

        print(f"Sample #{i+1} | Latency: {latency:.2f}s")
        print(f"PREDICT: {prediction[:100]}...")

    # 5. Calculate Final Metrics
    print("\n" + "="*25 + " FINAL PERFORMANCE METRICS " + "="*25)
    
    results = rouge.compute(predictions=predictions, references=references)
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"ROUGE-1: {results['rouge1']*100:.2f}")
    print(f"ROUGE-2: {results['rouge2']*100:.2f}")
    print(f"ROUGE-L: {results['rougeL']*100:.2f}")
    print(f"Avg Latency: {avg_latency:.2f} seconds per summary")
    print("=" * 77)

if __name__ == "__main__":
    run_inference(num_samples=10)
