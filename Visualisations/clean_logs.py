import json
import re
from pathlib import Path

def clean_logs():
    input_path = Path(__file__).parent / "training.txt"
    output_path = Path(__file__).parent / "cleaned_training_logs.json"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r") as f:
        lines = f.readlines()

    cleaned_data = []
    
    # Regex to catch dictionary-like strings: {'loss': '...', ...}
    dict_pattern = re.compile(r"\{'.*?\}")

    for line in lines:
        match = dict_pattern.search(line)
        if match:
            try:
                # Convert single quotes to double quotes for valid JSON
                # Be careful with nested strings if any (shouldn't be in these logs)
                raw_dict_str = match.group(0).replace("'", '"')
                data = json.loads(raw_dict_str)
                cleaned_data.append(data)
            except Exception as e:
                print(f"Failed to parse line: {line.strip()} | Error: {e}")

    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    print(f"Successfully cleaned logs. Found {len(cleaned_data)} entries.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    clean_logs()
