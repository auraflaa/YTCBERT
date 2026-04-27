import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# Setup paths
VIS_DIR = Path(__file__).resolve().parent
DATA_DIR = VIS_DIR.parent / "Model_Training"
CLEANED_LOGS = VIS_DIR / "cleaned_training_logs.json"
EVAL_RESULTS = VIS_DIR / "eval_results.json"
DATASET_FILE = DATA_DIR / "summarization_dataset.jsonl"

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (10, 6)
COLOR_PALETTE = sns.color_palette("muted")

def load_dataset_df():
    data = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df['input_word_count'] = df['input'].apply(lambda x: len(str(x).split()))
    df['output_word_count'] = df['output'].apply(lambda x: len(str(x).split()))
    return df

def generate_training_plots():
    if not CLEANED_LOGS.exists():
        print("Cleaned logs not found.")
        return

    with open(CLEANED_LOGS, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    for col in ['loss', 'eval_loss', 'learning_rate', 'epoch']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Training vs Validation Loss
    plt.figure()
    train_loss = df[df['loss'].notna()]
    eval_loss = df[df['eval_loss'].notna()]
    
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss', linewidth=2, color=COLOR_PALETTE[0])
    if not eval_loss.empty:
        plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Validation Loss', linewidth=3, color=COLOR_PALETTE[1])
    
    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(VIS_DIR / "training_val_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Learning Rate
    plt.figure()
    plt.plot(train_loss['epoch'], train_loss['learning_rate'], color=COLOR_PALETTE[2])
    plt.title('Learning Rate Schedule', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(VIS_DIR / "learning_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_data_plots():
    df = load_dataset_df()

    # 1. Length Distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df['input_word_count'], ax=axes[0], color=COLOR_PALETTE[0], kde=True)
    axes[0].set_title('Input Word Count Distribution')
    sns.histplot(df['output_word_count'], ax=axes[1], color=COLOR_PALETTE[1], kde=True)
    axes[1].set_title('Output Word Count Distribution')
    plt.tight_layout()
    plt.savefig(VIS_DIR / "data_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Top Keywords
    all_text = " ".join(df['input'].astype(str)).lower()
    words = re.findall(r'\b\w+\b', all_text)
    stop_words = {'the', 'a', 'to', 'and', 'i', 'is', 'in', 'it', 'of', 'for', 'you', 'this', 'that', 'with', 'on', 'was', 'my', 'is', 'are', 'was', 'were'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
    top_words = Counter(filtered_words).most_common(20)
    
    word_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    plt.figure(figsize=(10, 8))
    sns.barplot(data=word_df, y='Word', x='Count', palette='viridis')
    plt.title('Top 20 Keywords in Audience Comments', fontsize=16)
    plt.savefig(VIS_DIR / "data_keywords.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_plots():
    if not EVAL_RESULTS.exists():
        print("Evaluation results not found.")
        return

    with open(EVAL_RESULTS, "r") as f:
        data = json.load(f)
    samples_df = pd.DataFrame(data['samples'])

    # 1. ROUGE Boxplot
    rouge_data = samples_df[['rouge1', 'rouge2', 'rougeL']].melt(var_name='Metric', value_name='Score')
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=rouge_data, x='Metric', y='Score', palette='Set3')
    plt.title('Model Performance: ROUGE Score Distributions', fontsize=16)
    plt.ylabel('Score (0-100)')
    plt.ylim(0, 15) # Focused scale for 8.26 avg
    plt.savefig(VIS_DIR / "performance_rouge.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1.1 Semantic Similarity Boxplot
    if 'semantic_match' in samples_df.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=samples_df, y='semantic_match', color=COLOR_PALETTE[4])
        plt.title('Model Performance: Semantic Similarity', fontsize=16)
        plt.ylabel('Similarity Score (0.0 - 1.0)')
        plt.ylim(0, 1)
        plt.savefig(VIS_DIR / "performance_semantic.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Latency Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(samples_df['latency'], color=COLOR_PALETTE[3], kde=True)
    plt.title('Inference Latency Distribution (Device: CPU)', fontsize=16)
    plt.xlabel('Latency (seconds)')
    plt.savefig(VIS_DIR / "performance_latency.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Image Visualizations...")
    generate_training_plots()
    generate_data_plots()
    generate_performance_plots()
    print("Visualizations saved to Visualisations/ directory as PNG files.")
