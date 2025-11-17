# ui_components.py
# Streamlit-native UI only


def sentiment_badge(sentiment: str):
    """Return a simple colored label using Streamlit native text."""
    sentiment = sentiment.lower()
    if sentiment == "positive":
        return "🟢 Positive"
    elif sentiment == "negative":
        return "🔴 Negative"
    return "⚪ Neutral"


def aspect_list(aspects):
    """Return comma-separated aspects, safe for numpy arrays."""
    if aspects is None:
        return "None"

    # convert numpy array → python list
    try:
        aspects = list(aspects)
    except Exception:
        pass

    if len(aspects) == 0:
        return "None"

    return ", ".join(str(a) for a in aspects)
