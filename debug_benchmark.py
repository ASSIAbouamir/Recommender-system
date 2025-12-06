import pandas as pd
import numpy as np
from src.data_loader import AmazonDataLoader

def test_split_logic():
    print("Testing LOO Split Logic...")
    
    # Mock dataframe with time
    df = pd.DataFrame({
        'reviewerID': ['u1', 'u1', 'u1', 'u2', 'u2', 'u3'],
        'asin': ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'],
        'unixReviewTime': [100, 200, 300, 400, 500, 600]
    })
    
    print("Original Data:")
    print(df)
    
    user_counts = df['reviewerID'].value_counts()
    valid_users = user_counts[user_counts >= 3].index.tolist() # u1 only
    filtered_reviews = df[df['reviewerID'].isin(valid_users)].copy()
    
    print(f"\nFiltered (>=3 reviews) Users: {valid_users}")
    print(filtered_reviews)
    
    # Apply logic from run_experiments.py
    filtered_reviews['rank'] = filtered_reviews.groupby('reviewerID')['unixReviewTime'].rank(method='first', ascending=False)
    
    test_df = filtered_reviews[filtered_reviews['rank'] == 1]
    val_df = filtered_reviews[filtered_reviews['rank'] == 2]
    train_df = filtered_reviews[filtered_reviews['rank'] > 2]
    
    print("\nTest Rank == 1 (Should be u1, i3, t=300):")
    print(test_df[['reviewerID', 'asin', 'unixReviewTime', 'rank']])
    
    print("\nTrain Rank > 2 (Should be u1, i1, t=100):")
    print(train_df[['reviewerID', 'asin', 'unixReviewTime', 'rank']])
    
    assert len(test_df) == 1
    assert test_df.iloc[0]['asin'] == 'i3'
    assert len(train_df) == 1
    assert train_df.iloc[0]['asin'] == 'i1'
    
    print("\nSUCCESS: Split logic validated.")

if __name__ == "__main__":
    test_split_logic()
