import torch
import time
import json
import evaluate
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup paths
MT_DIR = Path(__file__).resolve().parent
FINAL_MODEL = MT_DIR / "YTCBERT"
DATASET_FILE = MT_DIR / "summarization_dataset.jsonl"
OUTPUT_FILE = MT_DIR.parent / "Visualisations" / "eval_results.json"

def get_hw_config():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            _ = torch.zeros(1, device="xpu")
            return "xpu"
        except: pass
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def run_evaluation(num_samples=10):
    if not FINAL_MODEL.exists():
        print("Model not found.")
        return

    device_type = "cpu"
    device = torch.device(device_type)
    print(">>> Running on: CPU (Forced for stability)")
    
    # 1. Load Metrics & Data
    rouge = evaluate.load("rouge")
    raw_dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    split_ds = raw_dataset.train_test_split(test_size=0.1, seed=42)
    test_set = split_ds["test"]
    
    max_test_samples = min(len(test_set), num_samples)
    samples = test_set.select(range(max_test_samples))

    # 2. Load Model
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL))
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(FINAL_MODEL),
        torch_dtype=torch.float32
    ).to(device)
    model.eval()

    detailed_results = []
    
    print(f"Starting evaluation on {max_test_samples} samples...")
    for i, example in enumerate(samples):
        raw_text = example["input"]
        reference = example["output"]
        
        clean_text = raw_text.replace("Summarize audience comments:", "").strip()
        input_text = f"summarize: {clean_text}"
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output_tokens = model.generate(**inputs, max_length=128, num_beams=4)
        latency = time.time() - start_time
        
        prediction = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # Calculate individual ROUGE for this sample
        sample_rouge = rouge.compute(predictions=[prediction], references=[reference])
        
        detailed_results.append({
            "sample_index": i,
            "latency": latency,
            "rouge1": sample_rouge["rouge1"],
            "rouge2": sample_rouge["rouge2"],
            "rougeL": sample_rouge["rougeL"],
            "input_len": len(clean_text),
            "output_len": len(reference),
            "pred_len": len(prediction)
        })
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{max_test_samples}...")

    # Calculate global averages
    avg_metrics = {
        "avg_rouge1": sum(r["rouge1"] for r in detailed_results) / len(detailed_results),
        "avg_rouge2": sum(r["rouge2"] for r in detailed_results) / len(detailed_results),
        "avg_rougeL": sum(r["rougeL"] for r in detailed_results) / len(detailed_results),
        "avg_latency": sum(r["latency"] for r in detailed_results) / len(detailed_results)
    }

    final_output = {
        "summary": avg_metrics,
        "samples": detailed_results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()
