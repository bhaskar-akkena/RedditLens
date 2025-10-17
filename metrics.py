"""
analyze_metrics.py
------------------
Generate deeper metrics for RedditLens Checkpoint 2 report:
- subreddit activity summary
- correlation heatmap
- temporal posting trends
- sentiment by subreddit
- lexical diversity
"""

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import Counter

sns.set_theme(style="whitegrid", font_scale=0.9)
plt.rcParams["figure.dpi"] = 150

DATA_PATH = Path("data/processed/cleaned.parquet")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_parquet(DATA_PATH)
print(f"Loaded {len(df)} posts from {df['subreddit'].nunique()} subreddits.")

# ---------- Sentiment (reuse) ----------
analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
df["sentiment"] = df["compound"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))

# ---------- 1. Subreddit summary ----------
summary = (
    df.groupby("subreddit")
    .agg(
        posts=("id", "count"),
        avg_score=("score", "mean"),
        avg_comments=("num_comments", "mean"),
        avg_len=("text_len", "mean"),
    )
    .round(2)
)
summary.to_csv("figures/subreddit_summary.csv")
print(summary)

# ---------- 2. Correlation heatmap ----------
plt.figure(figsize=(4,3))
corr = df[["score", "num_comments", "text_len", "compound"]].corr()
sns.heatmap(corr, annot=True, cmap="crest", fmt=".2f")
plt.title("Correlation Between Engagement, Length, and Sentiment")
plt.tight_layout()
plt.savefig(FIG_DIR / "6_corr_heatmap.png", bbox_inches="tight")
plt.close()

# ---------- 3. Temporal analysis ----------
df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
df["hour"] = df["created_dt"].dt.hour
hourly = df.groupby("hour")["id"].count()
plt.figure(figsize=(5,3))
sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="steelblue")
plt.title("Posting Activity by Hour (UTC)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Posts")
plt.tight_layout()
plt.savefig(FIG_DIR / "7_activity_by_hour.png", bbox_inches="tight")
plt.close()

# ---------- 4. Sentiment by subreddit ----------
sent_dist = pd.crosstab(df["subreddit"], df["sentiment"], normalize="index") * 100
sent_dist = sent_dist[["Positive", "Neutral", "Negative"]]
sent_dist.plot(kind="bar", stacked=True, figsize=(6,3), color=["#66c2a5", "#fc8d62", "#8da0cb"])
plt.title("Sentiment Composition by Subreddit")
plt.ylabel("Percentage of Posts")
plt.tight_layout()
plt.savefig(FIG_DIR / "8_sentiment_by_subreddit.png", bbox_inches="tight")
plt.close()

# ---------- 5. Lexical diversity ----------
def token_ratio(text):
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    if not tokens: return 0
    return len(set(tokens)) / len(tokens)

lex_div = df.groupby("subreddit")["text"].apply(lambda texts: np.mean([token_ratio(t) for t in texts]))
lex_div = lex_div.reset_index(name="lexical_diversity")
plt.figure(figsize=(5,3))
sns.barplot(y="subreddit", x="lexical_diversity", data=lex_div, palette="viridis")
plt.title("Lexical Diversity by Subreddit")
plt.xlabel("Unique Tokens / Total Tokens")
plt.ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "9_lexical_diversity.png", bbox_inches="tight")
plt.close()

print("✅ Extra figures saved in 'figures/' directory:")
for f in sorted(FIG_DIR.glob("6_*.png")):
    print(f.name)
for f in sorted(FIG_DIR.glob("7_*.png")):
    print(f.name)
for f in sorted(FIG_DIR.glob("8_*.png")):
    print(f.name)
for f in sorted(FIG_DIR.glob("9_*.png")):
    print(f.name)
