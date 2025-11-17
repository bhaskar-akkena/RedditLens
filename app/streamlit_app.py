import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import altair as alt

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "with_aspects.parquet")

df_all = pd.read_parquet(DATA_PATH)


# ------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------
df_all["title_preview"] = df_all["doc_text"].apply(lambda t: t.split(".")[0][:80] + "…")
df_all["snippet"] = df_all["doc_text"].apply(lambda x: x[:250].replace("\n", " ") + "…")


# ------------------------------------------------------
# TOKENIZER
# ------------------------------------------------------
def tokenize(text):
    return text.lower().split()


# ------------------------------------------------------
# BM25
# ------------------------------------------------------
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi(df_all["doc_text"].apply(tokenize).tolist())


# ------------------------------------------------------
# TF-IDF
# ------------------------------------------------------
tfidf = TfidfVectorizer().fit(df_all["doc_text"])
tfidf_matrix = tfidf.transform(df_all["doc_text"])

def tfidf_scores(query):
    return (tfidf_matrix @ tfidf.transform([query]).T).toarray().ravel()


# ------------------------------------------------------
# Reranker (MiniLM)
# ------------------------------------------------------
rerank_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = rerank_model.encode(df_all["doc_text"].tolist(), convert_to_tensor=True)

def rerank_scores(query):
    q = rerank_model.encode(query, convert_to_tensor=True)
    sims = util.cos_sim(q, embeddings)[0].cpu().numpy()
    return sims


# ------------------------------------------------------
# UTILITIES
# ------------------------------------------------------
def highlight(text, query):
    words = query.lower().split()
    pattern = "|".join(map(re.escape, words))
    if not pattern:
        return text
    return re.sub(f"({pattern})", r"**\1**", text, flags=re.IGNORECASE)


def sentiment_badge(s):
    return {"positive": "🟢 Positive", "negative": "🔴 Negative"}.get(s, "⚪ Neutral")


def aspect_list(a):
    if a is None:
        return "None"
    try:
        a = list(a)
    except:
        pass
    return "None" if len(a) == 0 else ", ".join(a)


def auto_subs(df, scores):
    top_idx = scores.argsort()[::-1][:30]
    return sorted(df.iloc[top_idx]["subreddit"].unique())


# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------

st.set_page_config(page_title="RedditLens", layout="wide")
st.title("RedditLens")

tab1, tab2, tab3 = st.tabs(["🔍 Search", "📊 Analytics", "📱 Product Insight"])

# Persist state
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ------------------------------------------------------
# SEARCH TAB
# ------------------------------------------------------
with tab1:

    st.header("Search Posts")

    query = st.text_input("Enter a query:", value=st.session_state.last_query)

    # Sorting
    sort_mode = st.selectbox(
        "Sort by:",
        ["BM25 Score", "TF-IDF Score", "Reranker Score", "Upvotes", "Comments", "Sentiment"],
    )

    # Run search button
    search_btn = st.button("Search")

    # Re-run based on preserved settings
    if search_btn:
        st.session_state.last_query = query

        # Compute scores
        scores_bm25 = bm25.get_scores(tokenize(query))
        scores_tfidf = tfidf_scores(query)
        scores_reranker = rerank_scores(query)

        # Save all scores
        df_all["bm25"] = scores_bm25
        df_all["tfidf"] = scores_tfidf
        df_all["rerank"] = scores_reranker

        # Auto-detected filters
        detected = auto_subs(df_all, scores_bm25)

        # Save initial search results
        st.session_state.last_results = df_all.copy()
        st.session_state.detected_subs = detected

    # Show results IF we have any
    if st.session_state.last_results is not None:

        df_results = st.session_state.last_results

        # Subreddit filter (persistent)
        subs = st.multiselect(
            "Filter by subreddit:",
            sorted(df_all["subreddit"].unique()),
            default=st.session_state.detected_subs,
            key="subs_filter"
        )

        df_results = df_results[df_results["subreddit"].isin(subs)]

        # Apply sorting
        if sort_mode == "BM25 Score":
            df_results = df_results.sort_values("bm25", ascending=False)
        elif sort_mode == "TF-IDF Score":
            df_results = df_results.sort_values("tfidf", ascending=False)
        elif sort_mode == "Reranker Score":
            df_results = df_results.sort_values("rerank", ascending=False)
        elif sort_mode == "Upvotes":
            df_results = df_results.sort_values("score", ascending=False)
        elif sort_mode == "Comments":
            df_results = df_results.sort_values("num_comments", ascending=False)
        elif sort_mode == "Sentiment":
            rank = {"positive": 2, "neutral": 1, "negative": 0}
            df_results["sent_rank"] = df_results["sentiment"].map(rank)
            df_results = df_results.sort_values("sent_rank", ascending=False)

        # Display results
        st.subheader(f"Showing {len(df_results)} results")

        for i, (_, row) in enumerate(df_results.head(30).iterrows(), start=1):

            with st.container():
                st.markdown(f"### {i}. {highlight(row['title_preview'], query)}")
                st.write(sentiment_badge(row["sentiment"]))
                st.write(highlight(row["snippet"], query))

                st.write(
                    f"**Subreddit:** r/{row['subreddit']}  \n"
                    f"**Upvotes:** {row['score']}  \n"
                    f"**Comments:** {row['num_comments']}  \n"
                    f"**BM25:** {row['bm25']:.2f}  \n"
                    f"**TF-IDF:** {row['tfidf']:.4f}  \n"
                    f"**Reranker:** {row['rerank']:.4f}"
                )

                st.write(f"**Aspects:** {aspect_list(row['aspects'])}")

                with st.expander("Full post"):
                    st.write(row["doc_text"])


# ------------------------------------------------------
# ANALYTICS TAB
# ------------------------------------------------------
with tab2:

    st.header("Aspect Importance Across Dataset")

    all_aspects = [a for row in df_all["aspects"] for a in list(row)]
    freq = pd.Series(all_aspects).value_counts().reset_index()
    freq.columns = ["aspect", "count"]

    chart = (
        alt.Chart(freq.head(25))
        .mark_bar()
        .encode(
            x="count:Q",
            y=alt.Y("aspect:N", sort="-x"),
        )
    )

    st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------
# PRODUCT INSIGHT TAB
# ------------------------------------------------------
with tab3:

    st.header("Product Insight")

    product_query = st.text_input(
        "Enter a product (e.g., iPhone 17, Dell XPS 15, Galaxy S24):",
        value="iPhone 17"
    )

    run_insight = st.button("Generate Product Insights")

    if run_insight:

        st.subheader(f"Insights for: **{product_query}**")

        # ------------------------------------------------------
        # RUN SEARCH OVER ENTIRE DATASET
        # ------------------------------------------------------
        scores_bm25 = bm25.get_scores(tokenize(product_query))
        scores_tfidf = tfidf_scores(product_query)
        scores_reranker = rerank_scores(product_query)

        df_temp = df_all.copy()
        df_temp["bm25"] = scores_bm25
        df_temp["tfidf"] = scores_tfidf
        df_temp["rerank"] = scores_reranker

        # Top 50 most relevant by reranker
        df_top = df_temp.sort_values("rerank", ascending=False).head(50)

        st.write(f"Found **{len(df_top)}** relevant posts.")

        # ------------------------------------------------------
        # OVERALL SENTIMENT
        # ------------------------------------------------------
        st.subheader("Overall Sentiment")

        sent_counts = df_top["sentiment"].value_counts()

        if len(sent_counts) > 0:
            df_sent = sent_counts.reset_index()
            df_sent.columns = ["sentiment", "count"]

            sent_chart = (
                alt.Chart(df_sent)
                .mark_bar()
                .encode(
                    x="sentiment:N",
                    y="count:Q",
                    tooltip=["sentiment", "count"]
                )
                .properties(width=400)
            )
            st.altair_chart(sent_chart, use_container_width=True)
        else:
            st.info("No sentiment data available for this product query.")

        # ------------------------------------------------------
        # ASPECT IMPORTANCE (PRODUCT SPECIFIC)
        # ------------------------------------------------------
        st.subheader("Top Aspects Mentioned")

        aspects = []
        for a_list in df_top["aspects"]:
            try:
                aspects.extend(list(a_list))
            except Exception:
                pass

        # default empty aspect_freq
        aspect_freq = pd.DataFrame(columns=["aspect", "count"])

        if len(aspects) == 0:
            st.warning("No aspects detected for this product.")
        else:
            aspect_freq = pd.Series(aspects).value_counts().reset_index()
            aspect_freq.columns = ["aspect", "count"]

            chart = (
                alt.Chart(aspect_freq.head(20))
                .mark_bar()
                .encode(
                    y=alt.Y("aspect:N", sort="-x"),
                    x="count:Q",
                    tooltip=["aspect", "count"]
                )
            )
            st.altair_chart(chart, use_container_width=True)

        # ------------------------------------------------------
        # ASPECT SENTIMENT TABLE
        # ------------------------------------------------------
        st.subheader("Aspect Sentiment Breakdown")

        rows = []
        for _, r in df_top.iterrows():
            for a in r["aspects"]:
                rows.append([a, r["sentiment"]])

        if len(rows) > 0:
            df_aspect = pd.DataFrame(rows, columns=["aspect", "sentiment"])
            aspect_sent = (
                df_aspect
                .groupby(["aspect", "sentiment"])
                .size()
                .unstack(fill_value=0)
            )
            st.dataframe(aspect_sent)
        else:
            st.info("No aspects found for sentiment table.")

        # ------------------------------------------------------
        # PRODUCT SUMMARY (DATA-DRIVEN)
        # ------------------------------------------------------
        st.subheader("Summary (Data-Driven)")

        if len(aspect_freq) > 0:
            top_aspects = aspect_freq.head(5)["aspect"].tolist()
        else:
            top_aspects = []

        summary_lines = []

        if top_aspects:
            summary_lines.append(f"- Most discussed aspects: **{', '.join(top_aspects)}**")

        if len(sent_counts) > 0:
            dominant = sent_counts.idxmax()
            summary_lines.append(f"- Overall sentiment leans **{dominant}**")
            summary_lines.append(
                f"- Sentiment counts → positive: {sent_counts.get('positive', 0)}, "
                f"negative: {sent_counts.get('negative', 0)}, "
                f"neutral: {sent_counts.get('neutral', 0)}"
            )
        else:
            summary_lines.append("- Not enough posts to estimate sentiment.")

        if not summary_lines:
            summary_lines.append("- No sufficient data to summarize this product.")

        st.markdown("\n".join(summary_lines))

        # ------------------------------------------------------
        # REPRESENTATIVE POSTS
        # ------------------------------------------------------
        st.subheader("Representative Posts")

        for i, (_, r) in enumerate(df_top.iterrows(), start=1):
            with st.expander(f"{i}. {r['title_preview']}"):
                st.write(r["doc_text"])
                st.write(f"**Sentiment:** {r['sentiment']}")
                st.write(f"**Aspects:** {aspect_list(r['aspects'])}")

