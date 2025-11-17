"""
clean_text.py (fixed)
Preserves readable Reddit text while removing URLs, emojis, and extra whitespace.
"""
import pandas as pd
import re, json, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Regular expressions
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
CTRL_RE = re.compile(r"[\r\n\t]+")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def clean_text(text):
    """Clean Reddit text while preserving readable content."""
    text = str(text)
    text = html.unescape(text)  # decode &amp;, &gt;, etc.
    text = re.sub(URL_RE, " ", text)
    text = re.sub(CTRL_RE, " ", text)
    # Remove only truly non-printable chars (not letters/numbers)
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    text = re.sub(WS_RE, " ", text)
    return text.strip()

def build_dataframe():
    rows = []
    for file in RAW_DIR.glob("*.jsonl"):
        print(f"📂 Processing {file.name}")
        for post in load_jsonl(file):
            comments = " ".join(post.get("comments", []))
            combined_text = f"{post.get('title','')} {post.get('selftext','')} {comments}"
            cleaned = clean_text(combined_text)
            if len(cleaned.split()) > 10:
                rows.append({
                    "id": post.get("id"),
                    "subreddit": post.get("subreddit"),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "doc_text": cleaned
                })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} cleaned documents.")
    df["text_len"] = df["doc_text"].apply(lambda x: len(x.split()))
    return df

if __name__ == "__main__":
    df = build_dataframe()
    df.to_parquet(OUT_PATH, index=False)
    print(f"💾 Saved cleaned dataset → {OUT_PATH}")
    print(df.head(3)[["subreddit", "doc_text"]])
