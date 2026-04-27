"""
Data Cleaning/processor.py
--------------------------
Core logic for the 8-step YouTube comment cleaning pipeline.
"""

import re
import html
import unicodedata
from difflib import SequenceMatcher

class CommentProcessor:
    def __init__(self, min_words=5, similarity_threshold=0.85, top_n=40):
        self.min_words = min_words
        self.similarity_threshold = similarity_threshold
        self.top_n = top_n

    def clean_single_comment(self, text):
        """Step 3, 4, 5: Normalization, Link Removal, Emoji Handling."""
        if not text:
            return ""

        # 1. Step 5 (Early): Initial decoding/unescaping
        text = html.unescape(text)
        
        # 2. Specific Escape Sequence Handling (JSON/C-style)
        text = text.replace("\\n", " ").replace("\\t", " ")
        text = text.replace('\\"', "'").replace("\\'", "'").replace('\\\\', '\\')
        
        # 3. Step 4: Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 4. Step 3: Remove Emojis & excessive symbols
        # We strip non-BMP characters (most emojis)
        text = "".join(c for c in text if unicodedata.category(c) != 'So')
        
        # 5. Normalization Phase 2: Quotes and Backslashes
        # Convert ALL double quotes to single quotes to prevent backslash-pollution in JSON
        text = text.replace('"', "'")
        # Remove any remaining lone backslashes
        text = text.replace('\\', '')
        
        # 6. Step 5 (Cont): Normalize whitespace and lowercase
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 7. Edge Case: Handling unclosed/dangling quotes
        # If it's a quote heavy string, we normalize. If it's just one at the start/end, we strip.
        if (text.startswith("'") or text.endswith("'")) and text.count("'") % 2 != 0:
            text = text.strip("'")

        # 8. Remove extra punctuation (optional but good)
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text

    def get_score(self, comment):
        """Step 7: Importance Scoring (Votes + Factor for favorited)."""
        try:
            votes = int(comment.get("votes", 0))
        except (ValueError, TypeError):
            votes = 0
            
        fav_bonus = 10 if comment.get("is_favorited") else 0
        return votes + fav_bonus

    def is_duplicate(self, text, existing_texts):
        """Step 6: Deduplication logic using SequenceMatcher."""
        for existing in existing_texts:
            similarity = SequenceMatcher(None, text.lower(), existing.lower()).ratio()
            if similarity > self.similarity_threshold:
                return True
        return False

    def process_comments(self, raw_comments):
        """Executes the full 8-step pipeline on a list of comments."""
        cleaned_pool = []
        
        # Steps 1-5 & 7
        for c in raw_comments:
            # Step 1: Remove replies
            if c.get("is_reply"):
                continue

            # Pre-clean the text for filtering
            raw_text = c.get("text", "")
            clean_text = self.clean_single_comment(raw_text)

            # Step 2: Remove low-length (< 5 words)
            words = clean_text.split()
            if len(words) < self.min_words:
                continue

            # Step 3 (Cont): Discard if emoji stripping left it too short
            if len(clean_text) < 5:
                continue

            # Store for batch processing
            cleaned_pool.append({
                "cid": c.get("cid"),
                "author": c.get("author"),
                "text": clean_text,
                "score": self.get_score(c),
                "original_votes": c.get("votes")
            })

        # Step 6: Deduplication (on the filtered pool)
        # We process from highest score down during dedup to keep the "best" version
        cleaned_pool.sort(key=lambda x: x["score"], reverse=True)
        
        deduplicated = []
        seen_texts = []
        
        for c in cleaned_pool:
            if not self.is_duplicate(c["text"], seen_texts):
                deduplicated.append(c)
                seen_texts.append(c["text"])
        
        # Step 8: Final Selection (Top 40)
        return deduplicated[:self.top_n]

if __name__ == "__main__":
    # Quick Test logic
    processor = CommentProcessor()
    test_data = [
        {"text": "This is a very informative video, thank you!", "votes": "10", "is_reply": False},
        {"text": "Informative video, thank you very much!", "votes": "5", "is_reply": False}, # Near duplicate
        {"text": "Nice", "votes": "100", "is_reply": False}, # Too short
        {"text": "Me too", "is_reply": True}, # Reply
        {"text": "Check this link: https://google.com", "votes": "1", "is_reply": False}, # Link
        {"text": "❤❤❤", "votes": "50", "is_reply": False}, # Emoji only
    ]
    results = processor.process_comments(test_data)
    print(f"Cleaned {len(test_data)} -> {len(results)}")
    for r in results:
        print(f"[{r['score']}] {r['text']}")
