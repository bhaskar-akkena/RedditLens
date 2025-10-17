"""
clean_text.py
-------------
Loads raw Reddit post data collected from data/raw/ (via public JSON endpoints),
cleans and merges it into a single DataFrame, and saves the processed dataset
to data/processed/cleaned.parquet.
"""

import re
import json
import pandas as pd
from pathlib import Path

# Directories
RAW_DIR = Path("../data/raw")
OUT_PATH = Path("../data/processed/cleaned.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Regex patterns for cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")

def load_jsonl_files(raw_dir: Path) -> pd.DataFrame:
    """Load and merge all JSONL subreddit files from data/raw/."""
    rows = []
    for file in raw_dir.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    post = json.loads(line)
                    text = f"{post.get('title', '')} {post.get('selftext', '')}".strip()
                    if text:
                        rows.append({
                            "id": post.get("id"),
                            "subreddit": post.get("subreddit"),
                            "title": post.get("title", ""),
                            "selftext": post.get("selftext", ""),
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "created_utc": post.get("created_utc", 0),
                            "text": text
                        })
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    print(f"✅ Loaded {len(df)} total posts from {len(list(raw_dir.glob('*.jsonl')))} files.")
    return df


def clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning (remove URLs, extra spaces, etc.)"""
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(URL_RE, " ", regex=True)
        .str.replace(MULTISPACE_RE, " ", regex=True)
        .str.strip()
    )
    df["text_len"] = df["text"].apply(lambda x: len(x.split()))
    df = df[df["text_len"] > 5]  # filter out extremely short entries
    print(f"✅ Cleaned dataset; remaining posts: {len(df)}")
    return df


def main():
    df = load_jsonl_files(RAW_DIR)
    df = clean_text_column(df)
    df.to_parquet(OUT_PATH, index=False)
    print(f"💾 Saved cleaned dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
