"""
collect_reddit.py
Collects Reddit posts and comments from selected subreddits
and saves them as JSONL files in data/raw/.
"""
import praw, os, json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# --- Load environment variables ---
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)
reddit.read_only = True

# --- Settings ---
SUBREDDITS = ["laptops", "SuggestALaptop", "smartphones", "Android", "iphone"]
LIMIT_POSTS = 400  # adjust based on how much data you want
OUTPUT_DIR = ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_subreddit(subreddit_name: str):
    print(f"\nCollecting from r/{subreddit_name} ...")
    data = []
    for post in reddit.subreddit(subreddit_name).new(limit=LIMIT_POSTS):
        post.comments.replace_more(limit=0)
        comments = [c.body for c in post.comments.list() if isinstance(c.body, str)]
        item = {
            "id": post.id,
            "subreddit": subreddit_name,
            "title": post.title,
            "selftext": post.selftext or "",
            "score": post.score,
            "num_comments": post.num_comments,
            "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat(),
            "comments": comments,
        }
        data.append(item)

    out_file = OUTPUT_DIR / f"{subreddit_name}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(data)} posts from r/{subreddit_name} → {out_file}")


if __name__ == "__main__":
    for sub in SUBREDDITS:
        collect_subreddit(sub)
    print("\nAll subreddits collected successfully!")
