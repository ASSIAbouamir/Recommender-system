
import pandas as pd
from pathlib import Path
import os

def check_data_stats():
    data_path = Path("data")
    reviews_file = data_path / "reviews.jsonl"
    processed_file = data_path / "Electronics_processed.parquet"

    print("Checking data files...")

    if reviews_file.exists():
        print(f"\nAnalyzing {reviews_file}...")
        try:
            # Read first few lines to check format since it's jsonl
            df = pd.read_json(reviews_file, lines=True)
            print(f"Total Rows: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
            if 'reviewerID' in df.columns:
                print(f"Unique Users: {df['reviewerID'].nunique()}")
            if 'asin' in df.columns:
                print(f"Unique Items: {df['asin'].nunique()}")
        except Exception as e:
            print(f"Error reading reviews: {e}")

    if processed_file.exists():
        print(f"\nAnalyzing {processed_file}...")
        try:
            df = pd.read_parquet(processed_file)
            print(f"Total Rows: {len(df)}")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading processed file: {e}")

if __name__ == "__main__":
    check_data_stats()
