"""
aspect_extraction.py
Extracts and analyzes common product aspects (e.g., battery, screen, performance)
from Reddit posts. Outputs an aspect frequency table and saves the enriched dataset.
"""

import pandas as pd
import re
from collections import Counter
from pathlib import Path
import spacy

# Load spaCy English model (download if needed)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_with_sentiment.parquet"
OUT_PATH = ROOT / "data" / "processed" / "with_aspects.parquet"

# Load dataset
print("Loading dataset with sentiment...")
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} documents.")

# Define aspect keywords / patterns
ASPECT_KEYWORDS = [
    "battery", "screen", "display", "keyboard", "camera", "performance",
    "speed", "heat", "fan", "noise", "build", "design", "port", "trackpad",
    "storage", "ssd", "ram", "processor", "cpu", "gpu", "graphics", "weight",
    "charger", "charging", "speaker", "audio", "price", "value"
]

ASPECT_RE = re.compile(r"\b(" + "|".join(ASPECT_KEYWORDS) + r")\b", flags=re.IGNORECASE)

# Extract aspects
print("Extracting aspects from text...")
def extract_aspects(text):
    found = ASPECT_RE.findall(text)
    return [f.lower() for f in found]

df["aspects"] = df["doc_text"].apply(extract_aspects)

# Count aspect frequencies
all_aspects = [a for sublist in df["aspects"] for a in sublist]
aspect_counts = Counter(all_aspects)
aspect_df = pd.DataFrame(aspect_counts.items(), columns=["aspect", "count"]).sort_values(by="count", ascending=False)

# Save enriched dataset
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"Saved dataset with aspects → {OUT_PATH}")

# Display top aspects
print("\nTop 15 most discussed aspects:")
print(aspect_df.head(15).to_string(index=False))

# Optional: sentiment breakdown per aspect
print("\nSentiment distribution for top aspects:")
aspect_sentiments = []
for aspect in aspect_df.head(10)["aspect"]:
    mask = df["aspects"].apply(lambda a_list: aspect in a_list)
    counts = df.loc[mask, "sentiment"].value_counts(normalize=True).mul(100).round(1)
    aspect_sentiments.append((aspect, dict(counts)))

for asp, sdict in aspect_sentiments:
    print(f"{asp:12s} → {sdict}")
