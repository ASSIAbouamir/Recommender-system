"""
Extract a sample of reviews from Tools_and_Home_Improvement dataset
"""
import json
from pathlib import Path
from tqdm import tqdm

input_file = Path("data/Tools_and_Home_Improvement.json/Tools_and_Home_Improvement.json")
output_file = Path("data/tools_reviews_sample.jsonl")

print(f"Reading from: {input_file}")
print(f"Writing to: {output_file}")

# Read JSONL file (one JSON object per line)
print("Loading JSONL file...")
all_reviews = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, desc="Reading lines")):
        if i >= 10000:  # Stop after 10k reviews
            break
        try:
            review = json.loads(line.strip())
            all_reviews.append(review)
        except json.JSONDecodeError as e:
            print(f"Skipping line {i}: {e}")
            continue

print(f"Loaded {len(all_reviews)} reviews")

# Write as JSONL
print(f"Writing to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for review in all_reviews:
        f.write(json.dumps(review) + '\n')

print(f"✅ Saved {len(all_reviews)} reviews to {output_file}")

# Get unique users and products
user_ids = set(r.get('reviewerID') for r in all_reviews if r.get('reviewerID'))
asins = set(r.get('asin') for r in all_reviews if r.get('asin'))

print(f"Unique users: {len(user_ids)}")
print(f"Unique products: {len(asins)}")
