"""
Batch Duplicate Removal for Video Lists.

Cleans video.txt by removing duplicate URLs while preserving order 
and category organization. Creates timestamped backups in the backups/ folder.
"""
import sys
from pathlib import Path
from datetime import datetime
from utils.helpers import extract_video_id

VIDEO_FILE = Path(__file__).parent / "video.txt"

def remove_duplicates(file_path):
    path = Path(file_path)
    if not path.exists():
        print(f"[ERR] File not found: {file_path}")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    
    unique_lines = []
    seen_ids = set()
    removed_count = 0

    for line in lines:
        clean = line.strip()
        
        # Keep comments and empty lines exactly as they are
        if not clean or clean.startswith("#"):
            unique_lines.append(line)
            continue
            
        # For actual video URLs/IDs
        video_id = extract_video_id(clean)
        if not video_id:
            # If we can't parse it, keep it as it might be an invalid URL for manual review
            unique_lines.append(line)
            continue
            
        if video_id in seen_ids:
            removed_count += 1
            continue
            
        seen_ids.add(video_id)
        unique_lines.append(line)

    if removed_count == 0:
        print("No duplicates found. The list is already clean.")
        return

    # Create a backup just in case
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}_{timestamp}.bak"
    
    backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    
    # Overwrite the original file
    path.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
    
    print(f"✅ Success! Removed {removed_count} duplicate link(s).")
    print(f"Original file backed up to: {backup_path}")
    print(f"Cleaned file: {path.name}")

if __name__ == "__main__":
    file_to_clean = sys.argv[1] if len(sys.argv) > 1 else VIDEO_FILE
    remove_duplicates(file_to_clean)
