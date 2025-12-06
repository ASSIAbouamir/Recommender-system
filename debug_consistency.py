import pandas as pd
import numpy as np
from src.data_loader import AmazonDataLoader
from baseline_cf import PopularityRecommender

def debug_consistency(category):
    print(f"Checking consistency for {category}...")
    loader = AmazonDataLoader(category=category, data_path="data")
    
    # Load raw data
    reviews_df = loader.load_reviews()
    meta_df = loader.load_metadata()
    full_df = loader.merge_and_preprocess()
    
    print(f"Total Products: {len(full_df)}")
    print(f"Total Reviews: {len(reviews_df)}")
    
    # Replicate filtering
    user_counts = reviews_df['reviewerID'].value_counts()
    valid_users = user_counts[user_counts >= 3].index.tolist()
    filtered_reviews = reviews_df[reviews_df['reviewerID'].isin(valid_users)].copy()
    
    print(f"Filtered Users (>=3 reviews): {len(valid_users)}")
    
    # Split logic
    filtered_reviews = filtered_reviews.sort_values('unixReviewTime')
    filtered_reviews['rank'] = filtered_reviews.groupby('reviewerID')['unixReviewTime'].rank(method='first', ascending=False)
    
    test_df = filtered_reviews[filtered_reviews['rank'] == 1]
    
    # Check if test items exist in metadata
    test_asins = test_df['asin'].unique()
    missing_asins = [a for a in test_asins if a not in full_df['asin'].values]
    
    print(f"Unique Test Items: {len(test_asins)}")
    print(f"Missing Test Items from Full DF: {len(missing_asins)}")
    if missing_asins:
        print(f"Example Missing: {missing_asins[:5]}")
        
    # Check Pop Recommender Coverage
    pop = PopularityRecommender(full_df)
    pop_top_100 = pop.popular_products.head(100)['asin'].values
    
    hits = 0
    for asin in test_asins:
        if asin in pop_top_100:
            hits += 1
            
    print(f"Test Items in Top 100 Popularity: {hits} / {len(test_asins)}")
    print(f"Expected Recall@100 (Popularity): {hits/len(test_asins):.4f}")

if __name__ == "__main__":
    debug_consistency("Tools_and_Home_Improvement")
