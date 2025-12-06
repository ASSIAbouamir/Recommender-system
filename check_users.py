import pandas as pd
from pathlib import Path

# Check if reviews file exists and has users
reviews_file = Path("data/tools_reviews_sample.jsonl")
if reviews_file.exists():
    df = pd.read_json(reviews_file, lines=True)
    users = df['reviewerID'].unique().tolist()[:20]
    print(f"Found {len(df['reviewerID'].unique())} unique users")
    print(f"First 20 users: {users}")
else:
    print(f"File not found: {reviews_file}")
