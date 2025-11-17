import praw, os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env in project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

print("CLIENT_ID:", os.getenv("CLIENT_ID"))
print("CLIENT_SECRET length:", len(os.getenv("CLIENT_SECRET") or ""))
print("USER_AGENT:", os.getenv("USER_AGENT"))

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)
reddit.read_only = True

try:
    print("Testing connection to subreddit 'laptops'...")
    subreddit = reddit.subreddit("laptops")
    print("✅ Success! Subreddit title:", subreddit.title)
except Exception as e:
    print("❌ Connection failed:", e)
