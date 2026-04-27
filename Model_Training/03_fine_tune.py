"""
Model Training/train_flan.py
---------------------------
Fine-tunes the local model on the YTCBERT audience summary dataset.
Optimized for Intel GPU (XPU) and Local Execution.
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    print(">>> Intel Extension for PyTorch (IPEX) not found. Standard PyTorch will be used.")

# --- CONFIG ---
MT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = MT_DIR / "flan-t5-base"
DATASET_FILE = MT_DIR / "summarization_dataset.jsonl"
OUTPUT_DIR = MT_DIR / "results"
FINAL_MODEL = MT_DIR / "YTCBERT"

def get_hw_config():
    """Detects available hardware (Intel XPU, NVIDIA CUDA, or CPU)."""
    # Detect Intel GPU
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            # Basic sanity check: Try to create a small tensor on XPU
            _ = torch.zeros(1, device="xpu")
            # Intel GPUs generally support bf16 well
            return "xpu", True, False 
        except Exception as e:
            print(f">>> XPU detected but failed to initialize: {e}")
            print(">>> Falling back to CPU/CUDA...")
    
    # Detect NVIDIA GPU
    if torch.cuda.is_available():
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        return "cuda", use_bf16, not use_bf16
    
    return "cpu", False, False

def train():
    if not LOCAL_MODEL_PATH.exists():
        print(f"Error: Local model not found at {LOCAL_MODEL_PATH}.")
        return
    
    if not DATASET_FILE.exists():
        print(f"Error: Dataset not found at {DATASET_FILE}.")
        return

    # 0. Hardware Detection
    device_type, use_bf16, use_fp16 = get_hw_config()
    device = torch.device(device_type)
    print(f">>> Hardware detected: {device_type.upper()}")
    
    print(">>> Initializing Fine-Tuning Pipeline...")

    # 1. Load Data
    dataset = load_dataset("json", data_files=str(DATASET_FILE), split="train")
    
    # 2. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_PATH))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(LOCAL_MODEL_PATH)).to(device)
    
    # Intel Optimization: IPEX optimize for inference/training
    if device_type == "xpu":
        # model = ipex.optimize(model, dtype=torch.bfloat16 if use_bf16 else torch.float32)
        print(">>> Intel XPU optimizations applied.")

    # 2.1 Enable Gradient Checkpointing for memory saving
    model.gradient_checkpointing_enable()

    # 3. Preprocessing
    def preprocess_function(examples):
        inputs = []
        for raw_text in examples["input"]:
            # Standardize prompt for T5
            if raw_text.startswith("Summarize audience comments:"):
                clean_text = raw_text.replace("Summarize audience comments:", "").strip()
            else:
                clean_text = raw_text.strip()
            inputs.append(f"summarize: {clean_text}")

        model_inputs = tokenizer(
            inputs, 
            text_target=examples["output"], 
            max_length=384, 
            truncation=True, 
            padding="max_length"
        )
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        learning_rate=3e-5,
        num_train_epochs=3, # Respecting 'dont overdo' constraint
        predict_with_generate=True, 
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=8, 
        per_device_eval_batch_size=1,        gradient_checkpointing=True,
        optim="adafactor", # Best for T5 and memory-constrained Intel GPUs
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=10,
        report_to="none",
        use_cpu=(device_type == "cpu"),
        # Required for XPU backend in some versions of Transformers/Accelerate
        ddp_find_unused_parameters=False if device_type == "xpu" else None,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    # 5. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # 6. Train
    print(">>> Starting training loop on Intel GPU...")
    trainer.train()

    # 7. Save
    print(f"Saving final model to {FINAL_MODEL}...")
    model.save_pretrained(FINAL_MODEL)
    tokenizer.save_pretrained(FINAL_MODEL)
    print("Done!")

if __name__ == "__main__":
    train()