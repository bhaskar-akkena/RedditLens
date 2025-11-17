# search_engine.py
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    return text.lower().split()


def build_bm25(df):
    tokenized_docs = df["doc_text"].apply(tokenize).tolist()
    return BM25Okapi(tokenized_docs)


def bm25_scores(bm25, df, query):
    return np.array(bm25.get_scores(tokenize(query)))


def subreddits_for_results(df, scores, top_k=30):
    idx = scores.argsort()[::-1][:top_k]
    return df.iloc[idx]["subreddit"].unique().tolist()


def aspect_sentiment_per_post(aspect, post_sentiment):
    """Simplified: use the post sentiment as aspect sentiment."""
    return post_sentiment.lower()
