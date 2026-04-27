"""
Model Inference/semantic_eval.py
---------------------------
Evaluates YTCBERT using Semantic Similarity (Cosine Similarity).
Better for capturing "meaning" rather than just word-overlap.
"""

import torch
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# --- CONFIG ---
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

def run_semantic_inference(num_samples=10):
    device_type = get_hw_config()
    device = torch.device(device_type)
    print(f">>> Evaluating on: {device_type.upper()}")

    # 1. Load Data & Semantic Model
    # 'all-MiniLM-L6-v2' is very fast and great for similarity tasks
    sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    raw_dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    
    # Use the same seed as training to get the holdout test set
    test_set = raw_dataset.train_test_split(test_size=0.1, seed=42)["test"]
    samples = test_set.select(range(min(len(test_set), num_samples)))

    # 2. Load YTCBERT
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(FINAL_MODEL)).to(device)
    model.eval()

    semantic_scores = []

    print("\n" + "═"*30 + " SEMANTIC EVALUATION " + "═"*30)

    for i, example in enumerate(samples):
        raw_text = example["input"]
        ground_truth = example["output"]

        # 3. Clean and Tokenize
        clean_text = raw_text.replace("Summarize audience comments:", "").strip()
        input_text = f"summarize: {clean_text}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=384, truncation=True).to(device)

        # 4. Generate
        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_length=96, num_beams=4)
        
        prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 5. Calculate Cosine Similarity
        # This converts both summaries into vectors and calculates the angle between them
        emb_pred = sim_model.encode(prediction, convert_to_tensor=True)
        emb_ref = sim_model.encode(ground_truth, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_pred, emb_ref).item()
        semantic_scores.append(similarity)

        print(f"\nSAMPLE #{i+1} | Semantic Match: {similarity:.2f}")
        print(f"GT:   {ground_truth[:80]}...")
        print(f"PRED: {prediction[:80]}...")

    # 6. Final Report
    avg_sim = sum(semantic_scores) / len(semantic_scores)
    print("\n" + "═"*25 + " FINAL SEMANTIC REPORT " + "═"*25)
    print(f"Average Semantic Similarity: {avg_sim:.4f}")
    print(f"Interpretation: {get_interpretation(avg_sim)}")
    print("═"*77)

def get_interpretation(score):
    if score > 0.85: return "Excellent - Meaning is nearly identical."
    if score > 0.70: return "Good - Captures core intent well."
    if score > 0.50: return "Fair - Some overlap, but misses nuances."
    return "Poor - Meaning is significantly different."

if __name__ == "__main__":
    run_semantic_inference()