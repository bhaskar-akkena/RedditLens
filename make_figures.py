"""
make_figures.py
---------------
Generate clean, publication-quality charts for Checkpoint 2 using real data.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# === Setup ===
DATA_PATH = Path("data/processed/cleaned.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150

# === Load data ===
df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df):,} documents from {df['subreddit'].nunique()} subreddits.")

# === Sentiment Analysis ===
analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(
    lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
)

# ---------------------------------------------------------------------------- #
# 1. Posts per subreddit
# ---------------------------------------------------------------------------- #
plt.figure(figsize=(6, 4))
sns.countplot(
    y="subreddit",
    data=df,
    order=df["subreddit"].value_counts().index,
    palette="crest"
)
plt.title("Posts Collected per Subreddit")
plt.xlabel("Number of Posts")
plt.ylabel("Subreddit")
plt.tight_layout()
plt.savefig(FIG_DIR / "1_posts_per_subreddit.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------- #
# 2. Document length distribution (fixed linear scale)
# ---------------------------------------------------------------------------- #
plt.figure(figsize=(6, 4))
sns.histplot(df[df["text_len"] < 600]["text_len"], bins=50, color="steelblue")
plt.title("Distribution of Document Lengths")
plt.xlabel("Words per Document")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig(FIG_DIR / "2_doc_length_distribution_fixed.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------- #
# 3. Sentiment distribution
# ---------------------------------------------------------------------------- #
sent_order = ["Positive", "Neutral", "Negative"]
sent_counts = df["sentiment"].value_counts(normalize=True).reindex(sent_order) * 100
plt.figure(figsize=(5.5, 4))
ax = sns.barplot(x=sent_counts.index, y=sent_counts.values, palette=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.title("Overall Sentiment Distribution (%)")
plt.ylabel("Percentage of Documents")
for i, v in enumerate(sent_counts.values):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "3_sentiment_distribution.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------- #
# 4. Average text length per subreddit
# ---------------------------------------------------------------------------- #
plt.figure(figsize=(6, 4))
avg_len = df.groupby("subreddit")["text_len"].mean().sort_values(ascending=False)
sns.barplot(x=avg_len.values, y=avg_len.index, palette="viridis")
plt.title("Average Document Length by Subreddit")
plt.xlabel("Average Words per Document")
plt.ylabel("Subreddit")
plt.tight_layout()
plt.savefig(FIG_DIR / "4_avg_length_per_subreddit.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------- #
# 5. Top frequent words (sampled corpus)
# ---------------------------------------------------------------------------- #
def tokenize(text):
    return [t.lower() for t in re.findall(r"\b[a-z]{3,}\b", text)]

sample = df.sample(min(1500, len(df)))["text"]
tokens = []
for t in sample:
    tokens.extend(tokenize(t))

freq = Counter(tokens)
common = pd.DataFrame(freq.most_common(20), columns=["term", "count"])

plt.figure(figsize=(6.5, 4.5))
sns.barplot(y="term", x="count", data=common, color="cornflowerblue")
plt.title("Top 20 Frequent Words (sample of corpus)")
plt.xlabel("Frequency")
plt.ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "5_top_terms.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------- #
# Summary console output
# ---------------------------------------------------------------------------- #
print("✅ Figures generated and saved in 'figures/' directory:")
for p in sorted(FIG_DIR.glob("*.png")):
    print("   •", p.name)
