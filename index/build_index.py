"""
build_index.py
Builds a BM25 index for RedditLens from the cleaned dataset and enables test queries.
"""

import pandas as pd
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Ensure tokenizer data is available
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
INDEX_PATH = ROOT / "data" / "processed" / "bm25_index.pkl"

# Load cleaned dataset
print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} documents.")

# Tokenize documents
print("Tokenizing text (this may take a minute)...")
tokenized_corpus = [word_tokenize(doc.lower()) for doc in df["doc_text"]]

# Build BM25 index
print("Building BM25 index...")
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 index built successfully!")

# Save index + dataframe for future use
with open(INDEX_PATH, "wb") as f:
    pickle.dump({"bm25": bm25, "df": df}, f)
print(f"Saved BM25 index to: {INDEX_PATH}")

# ---------------- TEST SEARCH ----------------
def search(query, k=5):
    tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = df.iloc[top_indices][["subreddit", "score", "doc_text"]].copy()
    results["bm25_score"] = [scores[i] for i in top_indices]
    return results

if __name__ == "__main__":
    # Example test query
    print("\nExample search for: 'battery life laptop'")
    results = search("battery life laptop", k=5)
    for i, row in results.iterrows():
        print(f"\n[{row['subreddit']}] (score={row['bm25_score']:.2f})")
        print(row["doc_text"][:300], "...")
