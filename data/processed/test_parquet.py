"""
verify_parquet.py
Quick inspection of the cleaned Reddit dataset (Parquet format).
"""
import pandas as pd
from pathlib import Path

# Path to processed dataset
DATA_PATH = Path("cleaned.parquet")

# Load data
df = pd.read_parquet(DATA_PATH)

# Display shape and basic info
print("Dataset loaded successfully!")
print(f"Documents: {len(df):,}")
print(f"Columns: {list(df.columns)}\n")

# Peek at first few rows
print("🔹 Sample rows:")
print(df.head(3)[["subreddit", "text_len", "doc_text"]])

# Basic stats
print("\nDocument length statistics:")
print(df["text_len"].describe())

# Count by subreddit
print("\nPosts per subreddit:")
print(df["subreddit"].value_counts())

# Check for nulls or empty strings
print("\nNull values per column:")
print(df.isna().sum())
