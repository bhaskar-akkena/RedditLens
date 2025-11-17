"""
sentiment_vader.py
Applies VADER sentiment analysis to the cleaned Reddit dataset
and saves the results for downstream visualization and ranking.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
OUT_PATH = ROOT / "data" / "processed" / "cleaned_with_sentiment.parquet"

# Load data
print("Loading cleaned dataset...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} documents.")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_label(score):
    """Convert compound score into categorical label."""
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis
print("Applying VADER sentiment analysis...")
df["compound"] = df["doc_text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(get_sentiment_label)

# Save enriched dataset
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"Saved dataset with sentiment → {OUT_PATH}")

# Display summary
print("\nSentiment distribution:")
print(df["sentiment"].value_counts(normalize=True).mul(100).round(2).astype(str) + " %")

# Show a few sample rows
print("\nSample rows with sentiment:")
print(df[["subreddit", "sentiment", "doc_text"]].head(3))
